"""Thin wrapper around ChatterboxTTS for use in the RunPod worker."""
from __future__ import annotations

import base64
import io
import logging
import os
import tempfile

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class ChatterboxRunner:
    """Loads the Chatterbox model once at worker startup and exposes generate()."""

    def __init__(self) -> None:
        import torch
        from chatterbox.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Chatterbox on device=%s", device)
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.sample_rate: int = self.model.sr
        logger.info("Chatterbox ready (sample_rate=%d)", self.sample_rate)

    def generate(
        self,
        text: str,
        reference_audio_b64: str,
        reference_text: str = "",
        language: str = "en",
        seed: int | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        **_kwargs,
    ) -> dict:
        """Generate audio and return base64-encoded WAV plus metadata."""
        import torch

        ref_bytes = base64.b64decode(reference_audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            ref_path = tmp.name

        try:
            generator = torch.Generator()
            if seed is not None:
                generator.manual_seed(seed)

            wav = self.model.generate(
                text,
                audio_prompt_path=ref_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )

            audio_np = wav.squeeze().cpu().numpy().astype(np.float32)
            duration_ms = int(len(audio_np) / self.sample_rate * 1000)

            buf = io.BytesIO()
            sf.write(buf, audio_np, self.sample_rate, format="WAV")
            audio_b64 = base64.b64encode(buf.getvalue()).decode()

            return {
                "audio_b64": audio_b64,
                "sample_rate": self.sample_rate,
                "duration_ms": duration_ms,
            }
        finally:
            os.unlink(ref_path)
