from __future__ import annotations

import networkx as nx

from loom.models import Node


class TopologicalSorter:
    """
    Stage 3 — Pure algorithmic stage. No LLM involved.

    Takes the validated nx.DiGraph from GraphBuilderValidator and
    produces execution levels via Kahn's algorithm (networkx
    topological_generations).

    Each level is a list of node IDs that:
    - Have no unresolved dependencies on each other
    - Can be executed fully in parallel

    The order of levels guarantees that when level N begins,
    all artifacts produced by levels 0..N-1 are available.
    """

    def sort(self, G: nx.DiGraph) -> list[list[str]]:
        """
        Produce topologically sorted execution levels.

        Args:
            G: Validated DAG from GraphBuilderValidator.

        Returns:
            List of levels, each level is a list of node IDs
            that can execute in parallel.

        Example:
            [
                ["data_ingestion"],                          # level 0
                ["market_researcher", "competitor_analyst"], # level 1 — parallel
                ["synthesizer_market_data"],                 # level 2
                ["strategy_agent"],                          # level 3
            ]
        """
        levels: list[list[str]] = []

        for generation in nx.topological_generations(G):
            levels.append(sorted(generation))  # sorted for deterministic ordering

        return levels

    def get_execution_order(self, G: nx.DiGraph) -> list[str]:
        """
        Flat ordered list of node IDs for debugging and logging.

        Args:
            G: Validated DAG from GraphBuilderValidator.

        Returns:
            Flat list of node IDs in execution order.
        """
        return [node for level in self.sort(G) for node in level]

    def describe(self, G: nx.DiGraph, nodes: dict[str, Node]) -> str:
        """
        Human readable description of the execution plan.
        Useful for logging and debugging.

        Args:
            G: Validated DAG.
            nodes: Node dict for name lookup.

        Returns:
            Multiline string describing each level and its agents.
        """
        levels = self.sort(G)
        lines: list[str] = ["Execution Plan:"]

        for i, level in enumerate(levels):
            agent_names = [nodes[node_id].name for node_id in level]
            parallel = " ‖ ".join(agent_names)
            lines.append(f"  Level {i}: [ {parallel} ]")

        return "\n".join(lines)