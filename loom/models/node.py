from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from .artifact import Artifact


class NodeType(str, Enum):
    """
    The role of a node in the DAG pipeline.
    """
    DECOMPOSER   = "decomposer"    # Meta — plans the DAG
    DOMAIN       = "domain"        # Does actual task work
    SYNTHESIZER  = "synthesizer"   # Auto-injected merge/QC node
    COMPILER     = "compiler"      # Final output assembly


class NodeStatus(str, Enum):
    """
    Execution lifecycle of a node.
    """
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"


class Node(BaseModel):
    """
    Represents a single agent in the DAG.

    Every node is an agent — decomposer, domain, synthesizer, and compiler
    all share this same interface. The node type determines behavior,
    but the structure is uniform.
    """

    id: str = Field(
        ...,
        description="Unique identifier for this node. Referenced by artifact contributors/users."
    )

    name: str = Field(
        ...,
        description="Human readable name for this agent node."
    )

    node_type: NodeType = Field(
        ...,
        description="Role of this node in the pipeline."
    )

    system_prompt: str = Field(
        ...,
        description="The system prompt that governs this agent's behavior and scope."
    )

    query_tool: dict[str, Any] = Field(
        default_factory=dict,
        description="ToolStorePy compatible tool configuration for this agent."
    )

    input_artifacts: list[str] = Field(
        default_factory=list,
        description="Names of artifacts this node consumes. Must be produced by upstream nodes."
    )

    output_artifacts: list[str] = Field(
        default_factory=list,
        description="Names of artifacts this node produces."
    )

    status: NodeStatus = Field(
        default=NodeStatus.PENDING,
        description="Current execution status of this node."
    )

    # Populated at runtime after execution
    result: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw LLM output after this node executes."
    )

    # Only populated for synthesizer nodes
    contributor_system_prompts: list[str] = Field(
        default_factory=list,
        description="System prompts of contributing agents. Used by synthesizer to understand contributor intent."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "market_researcher",
                "name": "Market Research Agent",
                "node_type": "domain",
                "system_prompt": "You are a market research specialist...",
                "query_tool": {},
                "input_artifacts": ["raw_data_summary"],
                "output_artifacts": ["market_analysis_report"],
                "status": "pending",
                "result": {},
                "contributor_system_prompts": []
            }
        }