"""TTS generation endpoints."""

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from .. import models
from ..services import history, profiles, tts
from ..database import Generation as DBGeneration, VoiceProfile as DBVoiceProfile, get_db
from ..services.generation import run_generation
from ..services.task_queue import enqueue_generation
from ..utils.tasks import get_task_manager

router = APIRouter()

# In-memory batch tracking: batch_id → ordered list of generation_ids
_batches: dict[str, list[str]] = {}


@router.post("/generate", response_model=models.GenerationResponse)
async def generate_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """Generate speech from text using a voice profile."""
    task_manager = get_task_manager()
    generation_id = str(uuid.uuid4())

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    from ..backends import engine_has_model_sizes

    engine = data.engine or "qwen"
    model_size = (data.model_size or "1.7B") if engine_has_model_sizes(engine) else None

    generation = await history.create_generation(
        profile_id=data.profile_id,
        text=data.text,
        language=data.language,
        audio_path="",
        duration=0,
        seed=data.seed,
        db=db,
        instruct=data.instruct,
        generation_id=generation_id,
        status="generating",
        engine=engine,
        model_size=model_size if engine_has_model_sizes(engine) else None,
    )

    task_manager.start_generation(
        task_id=generation_id,
        profile_id=data.profile_id,
        text=data.text,
    )

    effects_chain_config = None
    if data.effects_chain is not None:
        effects_chain_config = [e.model_dump() for e in data.effects_chain]
    else:
        import json as _json

        profile_obj = db.query(DBVoiceProfile).filter_by(id=data.profile_id).first()
        if profile_obj and profile_obj.effects_chain:
            try:
                effects_chain_config = _json.loads(profile_obj.effects_chain)
            except Exception:
                pass

    enqueue_generation(
        run_generation(
            generation_id=generation_id,
            profile_id=data.profile_id,
            text=data.text,
            language=data.language,
            engine=engine,
            model_size=model_size,
            seed=data.seed,
            normalize=data.normalize,
            effects_chain=effects_chain_config,
            instruct=data.instruct,
            mode="generate",
            max_chunk_chars=data.max_chunk_chars,
            crossfade_ms=data.crossfade_ms,
        )
    )

    return generation


@router.post("/generate/{generation_id}/retry", response_model=models.GenerationResponse)
async def retry_generation(generation_id: str, db: Session = Depends(get_db)):
    """Retry a failed generation using the same parameters."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if (gen.status or "completed") != "failed":
        raise HTTPException(status_code=400, detail="Only failed generations can be retried")

    gen.status = "generating"
    gen.error = None
    gen.audio_path = ""
    gen.duration = 0
    db.commit()
    db.refresh(gen)

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=gen.profile_id,
        text=gen.text,
    )

    enqueue_generation(
        run_generation(
            generation_id=generation_id,
            profile_id=gen.profile_id,
            text=gen.text,
            language=gen.language,
            engine=gen.engine or "qwen",
            model_size=gen.model_size or "1.7B",
            seed=gen.seed,
            instruct=gen.instruct,
            mode="retry",
        )
    )

    return models.GenerationResponse.model_validate(gen)


@router.post(
    "/generate/{generation_id}/regenerate",
    response_model=models.GenerationResponse,
)
async def regenerate_generation(generation_id: str, db: Session = Depends(get_db)):
    """Re-run TTS with the same parameters and save the result as a new version."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation must be completed to regenerate")

    gen.status = "generating"
    gen.error = None
    db.commit()
    db.refresh(gen)

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=gen.profile_id,
        text=gen.text,
    )

    version_id = str(uuid.uuid4())

    enqueue_generation(
        run_generation(
            generation_id=generation_id,
            profile_id=gen.profile_id,
            text=gen.text,
            language=gen.language,
            engine=gen.engine or "qwen",
            model_size=gen.model_size or "1.7B",
            seed=gen.seed,
            instruct=gen.instruct,
            mode="regenerate",
            version_id=version_id,
        )
    )

    return models.GenerationResponse.model_validate(gen)


@router.get("/generate/{generation_id}/status")
async def get_generation_status(generation_id: str, db: Session = Depends(get_db)):
    """SSE endpoint that streams generation status updates."""
    import json

    async def event_stream():
        try:
            while True:
                db.expire_all()
                gen = db.query(DBGeneration).filter_by(id=generation_id).first()
                if not gen:
                    yield f"data: {json.dumps({'status': 'not_found', 'id': generation_id})}\n\n"
                    return

                status = gen.status or "completed"
                payload = {
                    "id": gen.id,
                    "status": status,
                    "worker_status": status,
                    "duration": gen.duration,
                    "error": gen.error,
                }
                yield f"data: {json.dumps(payload)}\n\n"

                if (gen.status or "completed") in ("completed", "failed"):
                    return

                await asyncio.sleep(1)
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            logger.debug("SSE client disconnected for generation %s", generation_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/generate/stream")
async def stream_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """Generate speech and stream the WAV audio directly without saving to disk."""
    from ..backends import get_tts_backend_for_engine, ensure_model_cached_or_raise, load_engine_model, engine_needs_trim

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Mirror the regular /generate endpoint behavior more closely:
    # if the caller doesn't specify an engine, prefer the profile's default
    # engine (or preset engine) before falling back to qwen.
    engine = (
        data.engine
        or getattr(profile, "default_engine", None)
        or getattr(profile, "preset_engine", None)
        or "qwen"
    )
    tts_model = get_tts_backend_for_engine(engine)
    model_size = data.model_size or "1.7B"

    await ensure_model_cached_or_raise(engine, model_size)
    await load_engine_model(engine, model_size)

    voice_prompt = await profiles.create_voice_prompt_for_profile(
        data.profile_id,
        db,
        engine=engine,
    )

    from ..utils.chunked_tts import generate_chunked

    trim_fn = None
    if engine_needs_trim(engine):
        from ..utils.audio import trim_tts_output

        trim_fn = trim_tts_output

    audio, sample_rate = await generate_chunked(
        tts_model,
        data.text,
        voice_prompt,
        language=data.language,
        seed=data.seed,
        instruct=data.instruct,
        max_chunk_chars=data.max_chunk_chars,
        crossfade_ms=data.crossfade_ms,
        trim_fn=trim_fn,
    )

    effects_chain_config = None
    if data.effects_chain is not None:
        effects_chain_config = [e.model_dump() for e in data.effects_chain]
    elif profile.effects_chain:
        import json as _json

        try:
            effects_chain_config = _json.loads(profile.effects_chain)
        except Exception:
            effects_chain_config = None

    if effects_chain_config:
        from ..utils.effects import apply_effects

        audio = apply_effects(audio, sample_rate, effects_chain_config)

    if data.normalize:
        from ..utils.audio import normalize_audio

        audio = normalize_audio(audio)

    wav_bytes = tts.audio_to_wav_bytes(audio, sample_rate)

    async def _wav_stream():
        try:
            chunk_size = 64 * 1024
            for i in range(0, len(wav_bytes), chunk_size):
                yield wav_bytes[i : i + chunk_size]
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            logger.debug("Client disconnected during audio stream")

    return StreamingResponse(
        _wav_stream(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


@router.post("/generate/bulk", response_model=models.BulkGenerationResponse)
async def bulk_generate(
    data: models.BulkGenerationRequest,
    db: Session = Depends(get_db),
):
    """Submit multiple TTS jobs in parallel. Returns batch_id and per-job IDs."""
    if not data.texts:
        raise HTTPException(status_code=400, detail="texts cannot be empty")

    batch_id = str(uuid.uuid4())
    task_manager = get_task_manager()
    jobs = []

    for i, text in enumerate(data.texts):
        generation_id = str(uuid.uuid4())

        profile = await profiles.get_profile(data.profile_id, db)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile not found: {data.profile_id}")

        await history.create_generation(
            profile_id=data.profile_id,
            text=text,
            language=data.language,
            audio_path="",
            duration=0,
            seed=data.seed,
            db=db,
            instruct=None,
            generation_id=generation_id,
            status="queued",
            engine=data.engine,
            model_size=None,
        )

        task_manager.start_generation(
            task_id=generation_id,
            profile_id=data.profile_id,
            text=text,
        )

        enqueue_generation(
            run_generation(
                generation_id=generation_id,
                profile_id=data.profile_id,
                text=text,
                language=data.language,
                engine=data.engine,
                model_size=None,
                seed=data.seed,
                mode="generate",
            )
        )

        jobs.append(models.BulkJobStatus(
            job_id=generation_id,
            index=i,
            text=text,
            status="queued",
        ))

    _batches[batch_id] = [j.job_id for j in jobs]
    return models.BulkGenerationResponse(batch_id=batch_id, jobs=jobs)


@router.get("/generate/bulk/{batch_id}/status", response_model=models.BulkStatusResponse)
async def bulk_status(batch_id: str, db: Session = Depends(get_db)):
    """Poll the status of all jobs in a bulk batch."""
    generation_ids = _batches.get(batch_id)
    if not generation_ids:
        raise HTTPException(status_code=404, detail="Batch not found")

    jobs = []
    completed = 0
    failed = 0

    for i, gen_id in enumerate(generation_ids):
        gen = db.query(DBGeneration).filter_by(id=gen_id).first()
        if gen:
            status = gen.status or "completed"
            if status == "completed":
                completed += 1
            elif status == "failed":
                failed += 1
            jobs.append(models.BulkJobStatus(
                job_id=gen_id,
                index=i,
                text=gen.text,
                status=status,
                audio_path=gen.audio_path,
                error=gen.error,
            ))

    return models.BulkStatusResponse(
        batch_id=batch_id,
        total=len(generation_ids),
        completed=completed,
        failed=failed,
        jobs=jobs,
    )
