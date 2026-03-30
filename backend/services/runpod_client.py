"""RunPod serverless API client.

Three methods: submit_job, get_status, get_result.
All read RUNPOD_API_KEY and RUNPOD_GPU_ENDPOINT_ID from env.
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_API_KEY: str | None = None
_ENDPOINT_ID: str | None = None
_BASE_URL = "https://api.runpod.io/v2"


def _init() -> tuple[str, str]:
    global _API_KEY, _ENDPOINT_ID
    if _API_KEY is None:
        _API_KEY = os.environ["RUNPOD_API_KEY"]
    if _ENDPOINT_ID is None:
        _ENDPOINT_ID = os.environ["RUNPOD_GPU_ENDPOINT_ID"]
    return _API_KEY, _ENDPOINT_ID


def _headers() -> dict[str, str]:
    api_key, _ = _init()
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


async def submit_job(input_data: dict) -> str:
    """Submit a job to the GPU endpoint. Returns job_id."""
    _, endpoint_id = _init()
    url = f"{_BASE_URL}/{endpoint_id}/run"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json={"input": input_data}, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
    job_id = data["id"]
    logger.info("Submitted RunPod job %s", job_id)
    return job_id


async def get_status(job_id: str) -> dict:
    """Poll job status. Returns full RunPod status response dict."""
    _, endpoint_id = _init()
    url = f"{_BASE_URL}/{endpoint_id}/status/{job_id}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=_headers())
        resp.raise_for_status()
        return resp.json()


async def get_result(job_id: str) -> dict:
    """Return the output dict of a COMPLETED job. Raises if not completed."""
    data = await get_status(job_id)
    if data.get("status") != "COMPLETED":
        raise RuntimeError(
            f"Job {job_id} is not COMPLETED (status={data.get('status')})"
        )
    return data.get("output", {})
