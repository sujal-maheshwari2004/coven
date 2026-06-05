from __future__ import annotations

from coven.models import Artifact, Node, NodeType, NodeStatus
from .agent import GraphBuilderResponse, Edge


class GraphBuilderParser:
    """
    Converts the GraphBuilderResponse into updated Node and Artifact
    model instances, incorporating any repairs the LLM made.

    Also extracts synthesizer injection hints from the issues list.
    """

    def parse(
        self,
        response: GraphBuilderResponse,
    ) -> tuple[dict[str, Node], dict[str, Artifact], list[Edge], list[str]]:
        """
        Full parse of GraphBuilderResponse.

        Args:
            response: GraphBuilderResponse from GraphBuilderAgent.

        Returns:
            Tuple of:
            - nodes dict (corrected)
            - artifacts dict (corrected)
            - edges list
            - synthesizer_targets: artifact names needing synthesizer injection
        """
        nodes     = self._parse_nodes(response.nodes)
        artifacts = self._parse_artifacts(response.artifacts)
        edges     = response.edges
        synthesizer_targets = self._extract_synthesizer_targets(response.issues)

        return nodes, artifacts, edges, synthesizer_targets

    def _parse_nodes(self, raw_nodes: list[dict]) -> dict[str, Node]:
        nodes: dict[str, Node] = {}
        for raw in raw_nodes:
            node = Node(
                id=raw["id"],
                name=raw["name"],
                node_type=NodeType(raw["node_type"]),
                system_prompt=raw["system_prompt"],
                query_tool=raw.get("query_tool", {}),
                input_artifacts=raw.get("input_artifacts", []),
                output_artifacts=raw.get("output_artifacts", []),
                status=NodeStatus.PENDING,
            )
            nodes[node.id] = node
        return nodes

    def _parse_artifacts(self, raw_artifacts: list[dict]) -> dict[str, Artifact]:
        artifacts: dict[str, Artifact] = {}
        for raw in raw_artifacts:
            artifact = Artifact(
                name=raw["name"],
                description=raw["description"],
                contributors=raw.get("contributors", []),
                users=raw.get("users", []),
                body=raw.get("body", {}),
            )
            artifacts[artifact.name] = artifact
        return artifacts

    def _extract_synthesizer_targets(self, issues: list[str]) -> list[str]:
        """
        Scan the issues list for SYNTHESIZER_NEEDED hints emitted by the LLM.

        Returns:
            List of artifact names that need a synthesizer node injected.
        """
        targets: list[str] = []
        for issue in issues:
            if issue.startswith("SYNTHESIZER_NEEDED:"):
                artifact_name = issue.split(":", 1)[1].strip()
                targets.append(artifact_name)
        return targets