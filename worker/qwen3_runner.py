"""Thin wrapper around Qwen3TTSModel for use in the RunPod worker."""
from __future__ import annotations

import base64
import io
import logging
import os
import tempfile

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_LANGUAGE_CODE_TO_NAME: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
}


class Qwen3Runner:
    """Loads Qwen3-TTS once at worker startup and exposes generate()."""

    def __init__(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Qwen3-TTS on device=%s", device)

        model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        if device == "cpu":
            self.model = Qwen3TTSModel.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            )
        else:
            self.model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=torch.bfloat16,
            )

        logger.info("Qwen3-TTS ready")

    def _build_voice_prompt(self, ref_path: str, reference_text: str) -> dict:
        """Create voice clone prompt from a reference audio file."""
        return self.model.create_voice_clone_prompt(
            ref_audio=ref_path,
            ref_text=reference_text,
            x_vector_only_mode=False,
        )

    def generate(self, text: str, reference_audio_b64: str, **kwargs) -> dict:
        """Generate audio and return base64-encoded WAV plus metadata."""
        import torch

        reference_text: str = kwargs.get("reference_text", "")
        language: str = kwargs.get("language", "en")
        seed: int | None = kwargs.get("seed")
        instruct: str | None = kwargs.get("instruct")

        ref_bytes = base64.b64decode(reference_audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            ref_path = tmp.name

        try:
            if seed is not None:
                torch.manual_seed(seed)

            voice_prompt = self._build_voice_prompt(ref_path, reference_text)
            language_name = _LANGUAGE_CODE_TO_NAME.get(language, "auto")
            wavs, sample_rate = self.model.generate_voice_clone(
                text=text,
                voice_clone_prompt=voice_prompt,
                language=language_name,
                instruct=instruct,
            )

            audio_np = np.asarray(wavs[0], dtype=np.float32)
            duration_ms = int(len(audio_np) / sample_rate * 1000)

            buf = io.BytesIO()
            sf.write(buf, audio_np, sample_rate, format="WAV")
            return {
                "audio_b64": base64.b64encode(buf.getvalue()).decode(),
                "sample_rate": sample_rate,
                "duration_ms": duration_ms,
            }
        finally:
            os.unlink(ref_path)
