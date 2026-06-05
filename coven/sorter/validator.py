from __future__ import annotations

import networkx as nx

from coven.models import Node, Artifact


class SorterValidationError(Exception):
    """Raised when the graph fails pre-sort validation."""
    pass


class SorterValidator:
    """
    Pre-sort validation layer that runs immediately before topological sort.

    The GraphBuilderValidator already checked for cycles and unknown nodes.
    This validator focuses on runtime readiness:

    - Are all nodes present in the graph?
    - Does every node have at least one output artifact?
    - Are there any nodes referencing artifacts not yet in the artifact store?
    - Is the graph non-empty?

    Separating this from GraphBuilderValidator keeps each validator
    focused on its own stage's concerns.
    """

    def validate(
        self,
        G: nx.DiGraph,
        nodes: dict[str, Node],
        artifacts: dict[str, Artifact],
    ) -> None:
        """
        Run all pre-sort checks.

        Args:
            G: The validated DiGraph from GraphBuilderValidator.
            nodes: Node dict.
            artifacts: Artifact dict.

        Raises:
            SorterValidationError: On any pre-sort issue.
        """
        self._check_non_empty(G)
        self._check_all_nodes_in_graph(G, nodes)
        self._check_output_artifacts(nodes, artifacts)
        self._check_input_artifacts_exist(nodes, artifacts)

    def _check_non_empty(self, G: nx.DiGraph) -> None:
        if G.number_of_nodes() == 0:
            raise SorterValidationError(
                "DAG has no nodes. Nothing to execute."
            )

    def _check_all_nodes_in_graph(
        self,
        G: nx.DiGraph,
        nodes: dict[str, Node],
    ) -> None:
        graph_node_ids = set(G.nodes())
        model_node_ids = set(nodes.keys())

        missing_from_graph = model_node_ids - graph_node_ids
        if missing_from_graph:
            raise SorterValidationError(
                f"The following nodes exist in the node registry but are "
                f"missing from the DAG graph: {missing_from_graph}."
            )

        extra_in_graph = graph_node_ids - model_node_ids
        if extra_in_graph:
            raise SorterValidationError(
                f"The following node IDs exist in the DAG graph but have "
                f"no corresponding Node model: {extra_in_graph}."
            )

    def _check_output_artifacts(
        self,
        nodes: dict[str, Node],
        artifacts: dict[str, Artifact],
    ) -> None:
        """Every node must produce at least one artifact."""
        for node_id, node in nodes.items():
            if not node.output_artifacts:
                raise SorterValidationError(
                    f"Node '{node_id}' has no output artifacts. "
                    f"Every node must produce at least one artifact."
                )

    def _check_input_artifacts_exist(
        self,
        nodes: dict[str, Node],
        artifacts: dict[str, Artifact],
    ) -> None:
        """Every artifact referenced as input must exist in the artifact registry."""
        artifact_names = set(artifacts.keys())
        for node_id, node in nodes.items():
            for artifact_name in node.input_artifacts:
                if artifact_name not in artifact_names:
                    raise SorterValidationError(
                        f"Node '{node_id}' expects input artifact '{artifact_name}' "
                        f"which does not exist in the artifact registry."
                    )