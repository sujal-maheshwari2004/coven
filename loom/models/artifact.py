from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class Artifact(BaseModel):
    """
    Represents a unit of work passed between agents in the DAG.

    Artifacts are the edges of the DAG — the contributors and users
    fields define the graph structure implicitly.
    """

    name: str = Field(
        ...,
        description="Unique identifier for this artifact. Used to wire DAG edges."
    )

    description: str = Field(
        ...,
        description="Human and LLM readable description of what this artifact contains and represents."
    )

    contributors: list[str] = Field(
        default_factory=list,
        description="Agent node IDs that produce/write to this artifact."
    )

    users: list[str] = Field(
        default_factory=list,
        description="Agent node IDs that consume/read this artifact."
    )

    body: dict[str, Any] = Field(
        default_factory=dict,
        description="The actual artifact payload. Free-form JSON produced by contributors."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "market_analysis_report",
                "description": "Comprehensive analysis of target market size, segments, and growth trends.",
                "contributors": ["market_researcher"],
                "users": ["strategy_agent", "pitch_deck_agent"],
                "body": {
                    "market_size": "4.2B",
                    "segments": ["enterprise", "smb"],
                    "growth_rate": "12% YoY"
                }
            }
        }