from __future__ import annotations

from coven.models import Artifact, Node, NodeType, NodeStatus
from .agent import DecomposerResponse, DecomposedNode, DecomposedArtifact


class DecomposerParser:
    """
    Converts the raw DecomposerResponse from the LLM into
    validated Artifact and Node model instances.

    Keeps parsing logic separate from the LLM call so each
    can be tested and changed independently.
    """

    def parse_artifacts(
        self,
        raw_artifacts: list[DecomposedArtifact]
    ) -> dict[str, Artifact]:
        """
        Convert raw artifact dicts into Artifact models keyed by name.

        Args:
            raw_artifacts: List of DecomposedArtifact from LLM response.

        Returns:
            Dict of artifact name → Artifact model.
        """
        artifacts: dict[str, Artifact] = {}

        for raw in raw_artifacts:
            artifact = Artifact(
                name=raw.name,
                description=raw.description,
                contributors=raw.contributors,
                users=raw.users,
                body=raw.body,
            )
            artifacts[artifact.name] = artifact

        return artifacts

    def parse_nodes(
        self,
        raw_nodes: list[DecomposedNode]
    ) -> dict[str, Node]:
        """
        Convert raw node dicts into Node models keyed by node ID.

        Args:
            raw_nodes: List of DecomposedNode from LLM response.

        Returns:
            Dict of node ID → Node model.
        """
        nodes: dict[str, Node] = {}

        for raw in raw_nodes:
            node = Node(
                id=raw.id,
                name=raw.name,
                node_type=NodeType(raw.node_type),
                system_prompt=raw.system_prompt,
                query_tool=raw.query_tool,
                input_artifacts=raw.input_artifacts,
                output_artifacts=raw.output_artifacts,
                status=NodeStatus.PENDING,
            )
            nodes[node.id] = node

        return nodes

    def parse(self, response: DecomposerResponse) -> tuple[dict[str, Node], dict[str, Artifact]]:
        """
        Full parse of a DecomposerResponse into nodes and artifacts.

        Args:
            response: DecomposerResponse from DecomposerAgent.

        Returns:
            Tuple of (nodes dict, artifacts dict).
        """
        artifacts = self.parse_artifacts(response.artifacts)
        nodes     = self.parse_nodes(response.nodes)

        self._validate_references(nodes, artifacts)

        return nodes, artifacts

    def _validate_references(
        self,
        nodes: dict[str, Node],
        artifacts: dict[str, Artifact],
    ) -> None:
        """
        Ensure all artifact references in nodes point to real artifacts,
        and all contributor/user references in artifacts point to real nodes.

        Raises:
            ValueError: If any reference is broken.
        """
        artifact_names = set(artifacts.keys())
        node_ids       = set(nodes.keys())

        for node_id, node in nodes.items():
            for name in node.input_artifacts:
                if name not in artifact_names:
                    raise ValueError(
                        f"Node '{node_id}' references unknown input artifact '{name}'."
                    )
            for name in node.output_artifacts:
                if name not in artifact_names:
                    raise ValueError(
                        f"Node '{node_id}' references unknown output artifact '{name}'."
                    )

        for artifact_name, artifact in artifacts.items():
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