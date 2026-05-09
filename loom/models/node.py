from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class NodeType(str, Enum):
    DECOMPOSER   = "decomposer"
    DOMAIN       = "domain"
    SYNTHESIZER  = "synthesizer"
    COMPILER     = "compiler"


class NodeStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"


class ToolQuery(BaseModel):
    """
    A single tool description passed to ToolStorePy.
    Maps directly to one entry in queries.json format:
        {"tool_description": "evaluate a mathematical expression securely"}
    """
    tool_description: str = Field(
        ...,
        description=(
            "Plain English description of the tool this agent needs. "
            "ToolStorePy uses this to semantically search and build the right MCP tool."
        )
    )


class Node(BaseModel):
    """
    Represents a single agent in the DAG.

    Every node is an agent — decomposer, domain, synthesizer, and compiler
    all share this same interface. The node type determines behavior,
    but the structure is uniform.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "data_analyst",
                "name": "Data Analysis Agent",
                "node_type": "domain",
                "system_prompt": "You are a data analyst...",
                "query_tool": [
                    {"tool_description": "preview rows and get summary statistics from a CSV file"},
                ],
                "input_artifacts": ["raw_dataset"],
                "output_artifacts": ["analysis_report"],
                "status": "pending",
                "result": {},
                "contributor_system_prompts": [],
                "mcp_server_path": None,
            }
        }
    )

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

    query_tool: list[ToolQuery] = Field(
        default_factory=list,
        description=(
            "List of tool descriptions for ToolStorePy. "
            "Each entry describes one tool this agent needs in plain English. "
            "Leave empty if the agent needs no external tools."
        )
    )

    input_artifacts: list[str] = Field(
        default_factory=list,
        description="Names of artifacts this node consumes."
    )

    output_artifacts: list[str] = Field(
        default_factory=list,
        description="Names of artifacts this node produces."
    )

    status: NodeStatus = Field(
        default=NodeStatus.PENDING,
        description="Current execution status of this node."
    )

    result: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw LLM output after this node executes."
    )

    contributor_system_prompts: list[str] = Field(
        default_factory=list,
        description="System prompts of contributing agents. Used by synthesizer nodes."
    )

    mcp_server_path: str | None = Field(
        default=None,
        description="Absolute path to the ToolStorePy-built MCP server. Set at runtime."
    )