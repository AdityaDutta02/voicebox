"""Pydantic models for bulk generation requests and responses."""

from pydantic import BaseModel
from typing import Optional, List


class BulkGenerationRequest(BaseModel):
    """Request to generate audio for multiple texts in parallel."""

    texts: List[str]
    profile_id: str
    language: str = "en"
    engine: str = "qwen"
    seed: Optional[int] = None


class BulkJobStatus(BaseModel):
    """Status of a single job within a bulk batch."""

    job_id: str
    index: int
    text: str
    status: str
    audio_path: Optional[str] = None
    error: Optional[str] = None


class BulkGenerationResponse(BaseModel):
    """Response from POST /generate/bulk."""

    batch_id: str
    jobs: List[BulkJobStatus]


class BulkStatusResponse(BaseModel):
    """Response from GET /generate/bulk/{batch_id}/status."""

    batch_id: str
    total: int
    completed: int
    failed: int
    jobs: List[BulkJobStatus]
