"""RunPod serverless handler for Chatterbox TTS.

The ChatterboxRunner is instantiated once at module load (worker startup).
Subsequent jobs on a warm worker skip model loading entirely.
"""
import logging

import runpod

from qwen3_runner import Qwen3Runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing Qwen3-TTS model...")
runner = Qwen3Runner()
logger.info("Qwen3-TTS model ready.")


def handler(job: dict) -> dict:
    """Process a single TTS generation job."""
    inp = job["input"]
    logger.info(
        "Generating for profile_id=%s text_len=%d",
        inp.get("profile_id"),
        len(inp.get("text", "")),
    )
    result = runner.generate(**inp)
    result["profile_id"] = inp.get("profile_id")
    return result


runpod.serverless.start({"handler": handler})
