"""
tests/sorter/test_sorter.py

Unit tests for TopologicalSorter and SorterValidator.
Pure algorithmic — no LLM calls.
"""

import pytest
import networkx as nx

from loom.sorter.topological import TopologicalSorter
from loom.sorter.validator import SorterValidator, SorterValidationError
from loom.models import Node, NodeType, NodeStatus, Artifact


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _make_node(id, inputs=None, outputs=None):
    return Node(
        id=id,
        name=id.title(),
        node_type=NodeType.DOMAIN,
        system_prompt=f"You are {id}.",
        input_artifacts=inputs or [],
        output_artifacts=outputs or ["default_out"],
    )


def _make_artifact(name, contributors=None, users=None):
    return Artifact(
        name=name,
        description=f"Artifact {name}",
        contributors=contributors or [],
        users=users or [],
    )


def _linear_graph(node_ids: list[str]) -> nx.DiGraph:
    """Build a simple linear chain DAG."""
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    for i in range(len(node_ids) - 1):
        G.add_edge(node_ids[i], node_ids[i + 1])
    return G


def _parallel_graph() -> nx.DiGraph:
    """
    root → left  ─┐
                   → sink
    root → right ─┘
    """
    G = nx.DiGraph()
    G.add_nodes_from(["root", "left", "right", "sink"])
    G.add_edge("root", "left")
    G.add_edge("root", "right")
    G.add_edge("left", "sink")
    G.add_edge("right", "sink")
    return G


# ─────────────────────────────────────────────────────────────
# TOPOLOGICAL SORTER TESTS
# ─────────────────────────────────────────────────────────────

class TestTopologicalSorter:

    def setup_method(self):
        self.sorter = TopologicalSorter()

    def test_single_node(self):
        G = nx.DiGraph()
        G.add_node("agent_a")
        levels = self.sorter.sort(G)
        assert levels == [["agent_a"]]

    def test_linear_chain_produces_sequential_levels(self):
        G = _linear_graph(["a", "b", "c"])
        levels = self.sorter.sort(G)
        assert levels == [["a"], ["b"], ["c"]]

    def test_parallel_nodes_in_same_level(self):
        G = _parallel_graph()
        levels = self.sorter.sort(G)
        # root first, then left+right in parallel, then sink
        assert levels[0] == ["root"]
        assert sorted(levels[1]) == ["left", "right"]
        assert levels[2] == ["sink"]

    def test_levels_are_sorted_deterministically(self):
        G = nx.DiGraph()
        G.add_nodes_from(["z_node", "a_node"])
        levels = self.sorter.sort(G)
        # both isolated — same level, sorted alphabetically
        assert levels[0] == ["a_node", "z_node"]

    def test_all_nodes_appear_exactly_once(self):
        G = _parallel_graph()
        levels = self.sorter.sort(G)
        all_nodes = [n for level in levels for n in level]
        assert sorted(all_nodes) == sorted(G.nodes)
        assert len(all_nodes) == len(set(all_nodes))

    def test_get_execution_order_is_flat(self):
        G = _linear_graph(["a", "b", "c"])
        order = self.sorter.get_execution_order(G)
        assert order == ["a", "b", "c"]

    def test_get_execution_order_parallel(self):
        G = _parallel_graph()
        order = self.sorter.get_execution_order(G)
        # root must come first, sink must come last
        assert order[0] == "root"
        assert order[-1] == "sink"
        assert set(order[1:3]) == {"left", "right"}

    def test_describe_returns_string(self):
        G = _linear_graph(["a", "b"])
        nodes = {
            "a": _make_node("a"),
            "b": _make_node("b"),
        }
        desc = self.sorter.describe(G, nodes)
        assert isinstance(desc, str)
        assert "Level 0" in desc
        assert "Level 1" in desc

    def test_diamond_dag(self):
        """
        a → b ─┐
               → d
        a → c ─┘
        """
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
        levels = self.sorter.sort(G)
        assert levels[0] == ["a"]
        assert sorted(levels[1]) == ["b", "c"]
        assert levels[2] == ["d"]


# ─────────────────────────────────────────────────────────────
# SORTER VALIDATOR TESTS
# ─────────────────────────────────────────────────────────────

class TestSorterValidator:

    def setup_method(self):
        self.validator = SorterValidator()

    def _valid_setup(self):
        node_a = _make_node("agent_a", outputs=["art_1"])
        node_b = _make_node("agent_b", inputs=["art_1"], outputs=["art_2"])
        nodes = {"agent_a": node_a, "agent_b": node_b}
        artifacts = {
            "art_1": _make_artifact("art_1"),
            "art_2": _make_artifact("art_2"),
        }
        G = _linear_graph(["agent_a", "agent_b"])
        return G, nodes, artifacts

    def test_valid_setup_passes(self):
        G, nodes, artifacts = self._valid_setup()
        self.validator.validate(G, nodes, artifacts)  # should not raise

    def test_empty_graph_raises(self):
        G = nx.DiGraph()
        with pytest.raises(SorterValidationError, match="no nodes"):
            self.validator.validate(G, {}, {})

    def test_node_missing_from_graph_raises(self):
        node_a = _make_node("agent_a", outputs=["art_1"])
        nodes = {"agent_a": node_a, "ghost": _make_node("ghost", outputs=["x"])}
        artifacts = {"art_1": _make_artifact("art_1"), "x": _make_artifact("x")}
        G = nx.DiGraph()
        G.add_node("agent_a")
        # ghost is in nodes dict but not in G
        with pytest.raises(SorterValidationError, match="ghost"):
            self.validator.validate(G, nodes, artifacts)

    def test_extra_node_in_graph_raises(self):
        node_a = _make_node("agent_a", outputs=["art_1"])
        nodes = {"agent_a": node_a}
        artifacts = {"art_1": _make_artifact("art_1")}
        G = nx.DiGraph()
        G.add_nodes_from(["agent_a", "extra_node_in_graph"])
        with pytest.raises(SorterValidationError, match="extra_node_in_graph"):
            self.validator.validate(G, nodes, artifacts)

    def test_node_with_no_output_artifacts_raises(self):
        node_a = Node(
            id="agent_a",
            name="Agent A",
            node_type=NodeType.DOMAIN,
            system_prompt="...",
            output_artifacts=[],  # no outputs
        )
        nodes = {"agent_a": node_a}
        artifacts = {}
        G = nx.DiGraph()
        G.add_node("agent_a")
        with pytest.raises(SorterValidationError, match="no output artifacts"):
            self.validator.validate(G, nodes, artifacts)

    def test_input_artifact_not_in_registry_raises(self):
        node_a = _make_node("agent_a", inputs=["missing_artifact"], outputs=["art_1"])
        nodes = {"agent_a": node_a}
        artifacts = {"art_1": _make_artifact("art_1")}
        G = nx.DiGraph()
        G.add_node("agent_a")
        with pytest.raises(SorterValidationError, match="missing_artifact"):
            self.validator.validate(G, nodes, artifacts)