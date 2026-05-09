"""
tests/graph_builder/test_graph_builder.py

Unit tests for GraphBuilderParser and GraphBuilderValidator.
No LLM calls.
"""

import pytest
import networkx as nx

from loom.graph_builder.agent import GraphBuilderResponse, Edge
from loom.graph_builder.parser import GraphBuilderParser
from loom.graph_builder.validator import GraphBuilderValidator, GraphValidationError
from loom.models import Node, NodeType, NodeStatus, Artifact


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _node_dict(id, inputs=None, outputs=None):
    return {
        "id": id,
        "name": id.title(),
        "node_type": "domain",
        "system_prompt": f"You are {id}.",
        "query_tool": [],
        "input_artifacts": inputs or [],
        "output_artifacts": outputs or [],
    }


def _artifact_dict(name, contributors=None, users=None):
    return {
        "name": name,
        "description": f"Artifact {name}",
        "contributors": contributors or [],
        "users": users or [],
        "body": {},
    }


def _edge(from_node, to_node, artifact):
    return Edge(from_node=from_node, to_node=to_node, artifact=artifact)


def _response(nodes, artifacts, edges, issues=None):
    return GraphBuilderResponse(
        nodes=nodes,
        artifacts=artifacts,
        edges=edges,
        issues=issues or [],
    )


def _make_node(id, inputs=None, outputs=None):
    return Node(
        id=id,
        name=id.title(),
        node_type=NodeType.DOMAIN,
        system_prompt=f"You are {id}.",
        input_artifacts=inputs or [],
        output_artifacts=outputs or [],
    )


def _make_artifact(name, contributors=None, users=None):
    return Artifact(
        name=name,
        description=f"Artifact {name}",
        contributors=contributors or [],
        users=users or [],
    )


# ─────────────────────────────────────────────────────────────
# PARSER TESTS
# ─────────────────────────────────────────────────────────────

class TestGraphBuilderParser:

    def setup_method(self):
        self.parser = GraphBuilderParser()

    def test_parse_simple_graph(self):
        response = _response(
            nodes=[_node_dict("agent_a", outputs=["art_1"]),
                   _node_dict("agent_b", inputs=["art_1"])],
            artifacts=[_artifact_dict("art_1", contributors=["agent_a"], users=["agent_b"])],
            edges=[_edge("agent_a", "agent_b", "art_1")],
        )
        nodes, artifacts, edges, synth_targets = self.parser.parse(response)
        assert "agent_a" in nodes
        assert "agent_b" in nodes
        assert "art_1" in artifacts
        assert len(edges) == 1

    def test_parse_returns_node_models(self):
        response = _response(
            nodes=[_node_dict("agent_a", outputs=["art_1"])],
            artifacts=[_artifact_dict("art_1", contributors=["agent_a"])],
            edges=[],
        )
        nodes, _, _, _ = self.parser.parse(response)
        assert isinstance(nodes["agent_a"], Node)
        assert nodes["agent_a"].status == NodeStatus.PENDING

    def test_parse_returns_artifact_models(self):
        response = _response(
            nodes=[_node_dict("agent_a", outputs=["art_1"])],
            artifacts=[_artifact_dict("art_1", contributors=["agent_a"])],
            edges=[],
        )
        _, artifacts, _, _ = self.parser.parse(response)
        assert isinstance(artifacts["art_1"], Artifact)

    def test_extracts_synthesizer_targets_from_issues(self):
        response = _response(
            nodes=[_node_dict("agent_a", outputs=["art_1"])],
            artifacts=[_artifact_dict("art_1", contributors=["agent_a"])],
            edges=[],
            issues=["SYNTHESIZER_NEEDED: art_1", "Some other note"],
        )
        _, _, _, synth_targets = self.parser.parse(response)
        assert "art_1" in synth_targets
        assert len(synth_targets) == 1

    def test_ignores_non_synthesizer_issues(self):
        response = _response(
            nodes=[_node_dict("agent_a", outputs=["art_1"])],
            artifacts=[_artifact_dict("art_1", contributors=["agent_a"])],
            edges=[],
            issues=["Fixed dangling artifact", "Removed cycle"],
        )
        _, _, _, synth_targets = self.parser.parse(response)
        assert synth_targets == []

    def test_multiple_synthesizer_targets(self):
        response = _response(
            nodes=[],
            artifacts=[],
            edges=[],
            issues=[
                "SYNTHESIZER_NEEDED: artifact_x",
                "SYNTHESIZER_NEEDED: artifact_y",
            ],
        )
        _, _, _, synth_targets = self.parser.parse(response)
        assert "artifact_x" in synth_targets
        assert "artifact_y" in synth_targets

    def test_edges_preserved(self):
        edges = [
            _edge("a", "b", "art_1"),
            _edge("b", "c", "art_2"),
        ]
        response = _response(
            nodes=[_node_dict("a"), _node_dict("b"), _node_dict("c")],
            artifacts=[
                _artifact_dict("art_1", contributors=["a"], users=["b"]),
                _artifact_dict("art_2", contributors=["b"], users=["c"]),
            ],
            edges=edges,
        )
        _, _, parsed_edges, _ = self.parser.parse(response)
        assert len(parsed_edges) == 2


# ─────────────────────────────────────────────────────────────
# VALIDATOR TESTS
# ─────────────────────────────────────────────────────────────

class TestGraphBuilderValidator:

    def setup_method(self):
        self.validator = GraphBuilderValidator()

    def _simple_setup(self):
        nodes = {
            "agent_a": _make_node("agent_a", outputs=["art_1"]),
            "agent_b": _make_node("agent_b", inputs=["art_1"]),
        }
        artifacts = {
            "art_1": _make_artifact("art_1", contributors=["agent_a"], users=["agent_b"]),
        }
        edges = [_edge("agent_a", "agent_b", "art_1")]
        return nodes, artifacts, edges

    def test_valid_dag_returns_digraph(self):
        nodes, artifacts, edges = self._simple_setup()
        G = self.validator.validate(nodes, artifacts, edges)
        assert isinstance(G, nx.DiGraph)

    def test_valid_dag_has_correct_nodes(self):
        nodes, artifacts, edges = self._simple_setup()
        G = self.validator.validate(nodes, artifacts, edges)
        assert "agent_a" in G.nodes
        assert "agent_b" in G.nodes

    def test_valid_dag_has_correct_edges(self):
        nodes, artifacts, edges = self._simple_setup()
        G = self.validator.validate(nodes, artifacts, edges)
        assert G.has_edge("agent_a", "agent_b")

    def test_unknown_from_node_raises(self):
        nodes = {"agent_a": _make_node("agent_a")}
        artifacts = {}
        edges = [_edge("ghost_node", "agent_a", "art_1")]
        with pytest.raises(GraphValidationError, match="ghost_node"):
            self.validator.validate(nodes, artifacts, edges)

    def test_unknown_to_node_raises(self):
        nodes = {"agent_a": _make_node("agent_a")}
        artifacts = {}
        edges = [_edge("agent_a", "ghost_node", "art_1")]
        with pytest.raises(GraphValidationError, match="ghost_node"):
            self.validator.validate(nodes, artifacts, edges)

    def test_cycle_raises(self):
        nodes = {
            "agent_a": _make_node("agent_a"),
            "agent_b": _make_node("agent_b"),
        }
        artifacts = {}
        edges = [
            _edge("agent_a", "agent_b", "art_1"),
            _edge("agent_b", "agent_a", "art_2"),  # cycle
        ]
        with pytest.raises(GraphValidationError, match="cycle"):
            self.validator.validate(nodes, artifacts, edges)

    def test_isolated_node_raises(self):
        nodes = {
            "agent_a": _make_node("agent_a"),
            "agent_b": _make_node("agent_b"),  # isolated
        }
        artifacts = {}
        edges = []  # no edges — agent_b is isolated
        with pytest.raises(GraphValidationError, match="disconnected"):
            self.validator.validate(nodes, artifacts, edges)

    def test_linear_chain_is_valid(self):
        nodes = {
            "a": _make_node("a"),
            "b": _make_node("b"),
            "c": _make_node("c"),
        }
        artifacts = {}
        edges = [_edge("a", "b", "art_1"), _edge("b", "c", "art_2")]
        G = self.validator.validate(nodes, artifacts, edges)
        assert nx.is_directed_acyclic_graph(G)

    def test_parallel_nodes_valid(self):
        nodes = {
            "root":  _make_node("root"),
            "left":  _make_node("left"),
            "right": _make_node("right"),
            "sink":  _make_node("sink"),
        }
        artifacts = {}
        edges = [
            _edge("root",  "left",  "art_1"),
            _edge("root",  "right", "art_2"),
            _edge("left",  "sink",  "art_3"),
            _edge("right", "sink",  "art_4"),
        ]
        G = self.validator.validate(nodes, artifacts, edges)
        assert nx.is_directed_acyclic_graph(G)