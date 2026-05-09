from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field, model_validator

from .artifact import Artifact
from .node import Node, NodeStatus


class DAGStatus(str, Enum):
    """
    Overall execution state of the DAG.
    """
    PLANNED    = "planned"     # DAG built, not yet executing
    RUNNING    = "running"     # Currently executing levels
    COMPLETED  = "completed"   # All nodes completed successfully
    FAILED     = "failed"      # One or more nodes failed


from enum import Enum


class DAG(BaseModel):
    """
    The full pipeline — a directed acyclic graph of agent nodes
    connected by artifacts.

    Edges are implicit: an artifact's contributors → users defines
    the graph structure. No explicit edge list needed.

    Levels are populated by the topological sorter — each level
    contains nodes that can execute in parallel.
    """

    id: str = Field(
        ...,
        description="Unique identifier for this DAG run."
    )

    task: str = Field(
        ...,
        description="The original complex task this DAG was built to solve."
    )

    nodes: dict[str, Node] = Field(
        default_factory=dict,
        description="All agent nodes keyed by node ID."
    )

    artifacts: dict[str, Artifact] = Field(
        default_factory=dict,
        description="All artifacts keyed by artifact name."
    )

    levels: list[list[str]] = Field(
        default_factory=list,
        description="Topologically sorted execution levels. Each inner list is a set of node IDs that run in parallel."
    )

    status: DAGStatus = Field(
        default=DAGStatus.PLANNED,
        description="Current execution state of the DAG."
    )

    final_output: dict[str, Any] = Field(
        default_factory=dict,
        description="Compiled final output after all nodes complete."
    )

    @model_validator(mode="after")
    def validate_artifact_node_references(self) -> DAG:
        """
        Ensure every artifact contributor and user references
        a real node ID in the DAG.
        """
        node_ids = set(self.nodes.keys())

        for artifact_name, artifact in self.artifacts.items():
            for contributor in artifact.contributors:
                if contributor not in node_ids:
                    raise ValueError(
                        f"Artifact '{artifact_name}' contributor '{contributor}' "
                        f"does not match any node ID."
                    )
            for user in artifact.users:
                if user not in node_ids:
                    raise ValueError(
                        f"Artifact '{artifact_name}' user '{user}' "
                        f"does not match any node ID."
                    )

        return self

    def get_node(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in DAG.")
        return self.nodes[node_id]

    def get_artifact(self, artifact_name: str) -> Artifact:
        if artifact_name not in self.artifacts:
            raise KeyError(f"Artifact '{artifact_name}' not found in DAG.")
        return self.artifacts[artifact_name]

    def get_input_artifacts(self, node_id: str) -> list[Artifact]:
        """Return all artifact objects this node consumes."""
        node = self.get_node(node_id)
        return [self.get_artifact(name) for name in node.input_artifacts]

    def get_output_artifacts(self, node_id: str) -> list[Artifact]:
        """Return all artifact objects this node produces."""
        node = self.get_node(node_id)
        return [self.get_artifact(name) for name in node.output_artifacts]

    def is_complete(self) -> bool:
        """True when all nodes have completed successfully."""
        return all(
            node.status == NodeStatus.COMPLETED
            for node in self.nodes.values()
        )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "run_001",
                "task": "Produce a go-to-market strategy for a B2B SaaS product",
                "nodes": {},
                "artifacts": {},
                "levels": [["market_researcher"], ["strategy_agent", "pitch_deck_agent"]],
                "status": "planned",
                "final_output": {}
            }
        }