"""Pydantic v2 models for the Cerebra triage report.

All I/O boundaries use these models so that serialisation and validation are
centralised and type-safe.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class TriageDecision(str, Enum):
    FLAG = "FLAG"
    CLEAR = "CLEAR"


class Findings(BaseModel):
    """Per-study imaging findings extracted by the segmentation pipeline."""

    tumour_suspected: bool
    tumour_volume_mm3: float = Field(ge=0.0)
    anatomical_region: str
    max_diameter_mm: float = Field(ge=0.0)


class ModelMetadata(BaseModel):
    """Provenance for the model that produced this report."""

    model_id: str
    model_version: str
    inference_time_ms: int = Field(ge=0)
    device: str


class TriageReport(BaseModel):
    """Structured triage report returned by the Cerebra pipeline."""

    study_id: str
    triage_decision: TriageDecision
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    findings: Findings
    model_metadata: ModelMetadata
    visualization_path: str | None = None
    schema_version: str = "0.1.0"

    model_config = {"use_enum_values": True}
