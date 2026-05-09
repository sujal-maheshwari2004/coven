from __future__ import annotations

import networkx as nx

from loom.models import Artifact, Node
from .agent import Edge


class GraphValidationError(Exception):
    """Raised when the DAG fails structural validation."""
    pass


class GraphBuilderValidator:
    """
    Pure algorithmic validation of the DAG structure using networkx.

    Runs AFTER the LLM graph builder and parser have produced nodes,
    artifacts, and edges. This is the hard safety net — if the LLM
    missed a cycle or produced a disconnected graph, this catches it.

    Raises GraphValidationError on any structural problem.
    """

    def validate(
        self,
        nodes: dict[str, Node],
        artifacts: dict[str, Artifact],
        edges: list[Edge],
    ) -> nx.DiGraph:
        """
        Build and validate the networkx DiGraph.

        Args:
            nodes: Node dict from GraphBuilderParser.
            artifacts: Artifact dict from GraphBuilderParser.
            edges: Edge list from GraphBuilderParser.

        Returns:
            Valid nx.DiGraph ready for topological sort.

        Raises:
            GraphValidationError: On cycles, unknown node refs, or isolated nodes.
        """
        G = self._build_graph(nodes, edges)

        self._check_unknown_nodes(nodes, edges)
        self._check_cycles(G)
        self._check_isolated_nodes(G, nodes)

        return G

    def _build_graph(
        self,
        nodes: dict[str, Node],
        edges: list[Edge],
    ) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_nodes_from(nodes.keys())
        for edge in edges:
            G.add_edge(edge.from_node, edge.to_node, artifact=edge.artifact)
        return G

    def _check_unknown_nodes(
        self,
        nodes: dict[str, Node],
        edges: list[Edge],
    ) -> None:
        node_ids = set(nodes.keys())
        for edge in edges:
            if edge.from_node not in node_ids:
                raise GraphValidationError(
                    f"Edge references unknown from_node '{edge.from_node}'."
                )
            if edge.to_node not in node_ids:
                raise GraphValidationError(
                    f"Edge references unknown to_node '{edge.to_node}'."
                )

    def _check_cycles(self, G: nx.DiGraph) -> None:
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            raise GraphValidationError(
                f"DAG contains cycles: {cycles}. "
                f"The graph builder failed to resolve all circular dependencies."
            )

    def _check_isolated_nodes(
        self,
        G: nx.DiGraph,
        nodes: dict[str, Node],
    ) -> None:
        """
        Warn on nodes with no edges at all.
        These are either entry points with no dependencies (acceptable)
        or genuinely disconnected nodes (problematic).

        We allow nodes with no incoming edges (DAG roots) but flag
        nodes with neither incoming nor outgoing edges.
        """
        isolated = list(nx.isolates(G))
        if isolated:
            raise GraphValidationError(
                f"The following nodes have no edges at all and are disconnected "
                f"from the DAG: {isolated}. Every node must produce or consume "
                f"at least one artifact."
            )