"""
Unified TTS generation orchestration.

Replaces the three near-identical closures (_run_generation, _run_retry,
_run_regenerate) that lived in main.py with a single ``run_generation()``
function parameterized by *mode*.

Mode differences:
  - "generate"   : full pipeline -- save clean version, optionally apply
                    effects and create a processed version.
  - "retry"      : re-runs a failed generation with the same seed.
                    No effects, no version creation.
  - "regenerate" : re-runs with seed=None for variation.  Creates a new
                    version with an auto-incremented "take-N" label.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Literal, Optional

from .. import config
from . import history, profiles
from ..database import get_db
from ..utils.tasks import get_task_manager

# In-memory map: generation_id → RunPod job_id and submit time
_runpod_jobs: dict[str, str] = {}
_runpod_submit_times: dict[str, float] = {}


@dataclass
class _RunOpts:
    """Options shared between local and RunPod generation paths."""

    mode: Literal["generate", "retry", "regenerate"]
    seed: Optional[int] = None
    normalize: bool = False
    effects_chain: Optional[list] = field(default=None)
    instruct: Optional[str] = None
    max_chunk_chars: Optional[int] = None
    crossfade_ms: Optional[int] = None
    version_id: Optional[str] = None


def get_runpod_job_id(generation_id: str) -> str | None:
    """Return the RunPod job_id for a generation, or None if not dispatched."""
    return _runpod_jobs.get(generation_id)


async def run_generation(
    *,
    generation_id: str,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str,
    seed: Optional[int],
    normalize: bool = False,
    effects_chain: Optional[list] = None,
    instruct: Optional[str] = None,
    mode: Literal["generate", "retry", "regenerate"],
    max_chunk_chars: Optional[int] = None,
    crossfade_ms: Optional[int] = None,
    version_id: Optional[str] = None,
) -> None:
    """Execute TTS inference and persist the result.

    Routes to RunPod when RUNPOD_GPU_ENDPOINT_ID is set and engine is qwen or
    chatterbox, otherwise runs the local TTS pipeline.
    """
    opts = _RunOpts(
        mode=mode,
        seed=seed,
        normalize=normalize,
        effects_chain=effects_chain,
        instruct=instruct,
        max_chunk_chars=max_chunk_chars,
        crossfade_ms=crossfade_ms,
        version_id=version_id,
    )

    runpod_engines = ("qwen", "chatterbox", "chatterbox_turbo")
    if os.environ.get("RUNPOD_GPU_ENDPOINT_ID") and engine in runpod_engines:
        await _run_generation_runpod(
            generation_id=generation_id,
            profile_id=profile_id,
            text=text,
            language=language,
            engine=engine,
            opts=opts,
        )
        return

    from ..backends import load_engine_model, get_tts_backend_for_engine, engine_needs_trim
    from ..utils.chunked_tts import generate_chunked
    from ..utils.audio import trim_tts_output

    task_manager = get_task_manager()
    bg_db = next(get_db())

    try:
        tts_model = get_tts_backend_for_engine(engine)

        if not tts_model.is_loaded():
            await history.update_generation_status(generation_id, "loading_model", bg_db)

        await load_engine_model(engine, model_size)

        voice_prompt = await profiles.create_voice_prompt_for_profile(
            profile_id,
            bg_db,
            use_cache=True,
            engine=engine,
        )

        await history.update_generation_status(generation_id, "generating", bg_db)
        trim_fn = trim_tts_output if engine_needs_trim(engine) else None

        gen_kwargs: dict = dict(
            language=language,
            seed=seed if mode != "regenerate" else None,
            instruct=instruct,
            trim_fn=trim_fn,
        )
        if max_chunk_chars is not None:
            gen_kwargs["max_chunk_chars"] = max_chunk_chars
        if crossfade_ms is not None:
            gen_kwargs["crossfade_ms"] = crossfade_ms

        audio, sample_rate = await generate_chunked(tts_model, text, voice_prompt, **gen_kwargs)
        await _complete_audio(generation_id, audio, sample_rate, opts, bg_db)

    except Exception as e:
        traceback.print_exc()
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error=str(e),
        )
    finally:
        task_manager.complete_generation(generation_id)
        bg_db.close()


async def _complete_audio(generation_id, audio, sample_rate, opts: _RunOpts, db) -> None:
    """Normalize if needed, save, and mark generation completed."""
    from ..utils.audio import normalize_audio

    if opts.normalize or opts.mode == "regenerate":
        audio = normalize_audio(audio)

    duration = len(audio) / sample_rate
    final_path = _dispatch_save(generation_id, audio, sample_rate, opts, db)

    await history.update_generation_status(
        generation_id=generation_id,
        status="completed",
        db=db,
        audio_path=final_path,
        duration=duration,
    )


def _dispatch_save(generation_id, audio, sample_rate, opts: _RunOpts, db) -> str:
    """Call the right _save_* function based on opts.mode. Returns final path."""
    if opts.mode == "generate":
        return _save_generate(generation_id, audio, sample_rate, opts.effects_chain, db)
    if opts.mode == "retry":
        return _save_retry(generation_id, audio, sample_rate)
    return _save_regenerate(generation_id, opts.version_id, audio, sample_rate, db)


def _save_generate(generation_id, audio, sample_rate, effects_chain, db) -> str:
    """Save clean version and optionally an effects-processed version.

    Returns the final audio path (processed if effects were applied,
    otherwise clean).
    """
    from ..utils.audio import save_audio
    from . import versions as versions_mod

    clean_audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(clean_audio_path), sample_rate)

    has_effects = effects_chain and any(e.get("enabled", True) for e in effects_chain)

    versions_mod.create_version(
        generation_id=generation_id,
        label="original",
        audio_path=str(clean_audio_path),
        db=db,
        effects_chain=None,
        is_default=not has_effects,
    )

    if not has_effects:
        return str(clean_audio_path)

    return _apply_effects_version(generation_id, audio, sample_rate, effects_chain, db)


def _apply_effects_version(generation_id, audio, sample_rate, effects_chain, db) -> str:
    """Apply effects chain and save as processed version. Returns final path."""
    from ..utils.effects import apply_effects, validate_effects_chain
    from ..utils.audio import save_audio
    from . import versions as versions_mod
    import logging

    clean_audio_path = str(config.get_generations_dir() / f"{generation_id}.wav")
    error_msg = validate_effects_chain(effects_chain)
    if error_msg:
        logging.getLogger(__name__).warning("invalid effects chain, skipping: %s", error_msg)
        versions_mod.set_default_version(
            versions_mod.list_versions(generation_id, db)[0].id, db
        )
        return clean_audio_path

    processed_audio = apply_effects(audio, sample_rate, effects_chain)
    processed_path = config.get_generations_dir() / f"{generation_id}_processed.wav"
    save_audio(processed_audio, str(processed_path), sample_rate)
    versions_mod.create_version(
        generation_id=generation_id,
        label="version-2",
        audio_path=str(processed_path),
        db=db,
        effects_chain=effects_chain,
        is_default=True,
    )
    return str(processed_path)


def _save_retry(generation_id, audio, sample_rate) -> str:
    """Save retry output -- single file, no versions. Returns the audio path."""
    from ..utils.audio import save_audio

    audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(audio_path), sample_rate)
    return str(audio_path)


def _save_regenerate(generation_id, version_id, audio, sample_rate, db) -> str:
    """Save regeneration output as a new version with auto-label. Returns path."""
    from ..utils.audio import save_audio
    from . import versions as versions_mod
    import uuid as _uuid
    from ..database import GenerationVersion as DBGenerationVersion

    suffix = _uuid.uuid4().hex[:8]
    audio_path = config.get_generations_dir() / f"{generation_id}_{suffix}.wav"
    save_audio(audio, str(audio_path), sample_rate)

    count = db.query(DBGenerationVersion).filter_by(generation_id=generation_id).count()
    label = f"take-{count + 1}"

    versions_mod.create_version(
        generation_id=generation_id,
        label=label,
        audio_path=str(audio_path),
        db=db,
        effects_chain=None,
        is_default=True,
    )
    return str(audio_path)


async def _run_generation_runpod(
    *,
    generation_id: str,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    opts: _RunOpts,
) -> None:
    """Dispatch TTS generation to RunPod GPU worker and poll until complete."""
    import numpy as np
    import soundfile as sf

    from .runpod_client import submit_job, get_status
    from ..database import ProfileSample as DBProfileSample

    task_manager = get_task_manager()
    bg_db = next(get_db())

    try:
        samples = bg_db.query(DBProfileSample).filter_by(profile_id=profile_id).all()
        if not samples:
            raise ValueError(f"No audio samples for profile {profile_id}")

        sample = samples[0]
        with open(sample.audio_path, "rb") as f:
            ref_audio_b64 = base64.b64encode(f.read()).decode()

        input_data: dict = {
            "text": text,
            "reference_audio_b64": ref_audio_b64,
            "reference_text": sample.reference_text,
            "language": language,
            "profile_id": profile_id,
            "seed": opts.seed if opts.mode != "regenerate" else None,
        }
        if opts.instruct:
            input_data["instruct"] = opts.instruct

        job_id = await submit_job(input_data)
        _runpod_jobs[generation_id] = job_id
        _runpod_submit_times[generation_id] = time.time()

        await history.update_generation_status(generation_id, "queued", bg_db)

        while True:
            await asyncio.sleep(2)
            status_data = await get_status(job_id)
            rp_status = status_data.get("status", "IN_QUEUE")

            if rp_status == "IN_QUEUE":
                elapsed = time.time() - _runpod_submit_times[generation_id]
                db_status = "warming" if elapsed > 5 else "queued"
                await history.update_generation_status(generation_id, db_status, bg_db)

            elif rp_status == "IN_PROGRESS":
                await history.update_generation_status(generation_id, "generating", bg_db)

            elif rp_status == "COMPLETED":
                output = status_data.get("output", {})
                audio_bytes = base64.b64decode(output["audio_b64"])
                audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                audio_array = np.asarray(audio_array, dtype=np.float32)
                await _complete_audio(generation_id, audio_array, sr, opts, bg_db)
                break

            elif rp_status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                error_msg = status_data.get("error") or f"RunPod job {rp_status}"
                await history.update_generation_status(
                    generation_id=generation_id,
                    status="failed",
                    db=bg_db,
                    error=error_msg,
                )
                break

    except Exception as exc:
        traceback.print_exc()
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error=str(exc),
        )
    finally:
        _runpod_jobs.pop(generation_id, None)
        _runpod_submit_times.pop(generation_id, None)
        task_manager.complete_generation(generation_id)
        bg_db.close()
