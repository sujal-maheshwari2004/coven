"""
tests/models/test_models.py

Unit tests for loom/models/ — Artifact, Node, DAG, DAGStatus, NodeType, NodeStatus.
No LLM calls. Pure Pydantic validation.
"""

import pytest
from pydantic import ValidationError

from loom.models import Artifact, Node, NodeType, NodeStatus, DAG, DAGStatus
from loom.models.node import ToolQuery


# ─────────────────────────────────────────────────────────────
# ARTIFACT
# ─────────────────────────────────────────────────────────────

class TestArtifact:

    def test_basic_creation(self):
        a = Artifact(
            name="market_report",
            description="Market analysis output",
            contributors=["agent_a"],
            users=["agent_b"],
        )
        assert a.name == "market_report"
        assert a.body == {}

    def test_defaults(self):
        a = Artifact(name="x", description="y")
        assert a.contributors == []
        assert a.users == []
        assert a.body == {}

    def test_body_is_free_form(self):
        a = Artifact(
            name="x",
            description="y",
            body={"key": "value", "nested": {"a": 1}},
        )
        assert a.body["nested"]["a"] == 1

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            Artifact(description="no name")

    def test_missing_description_raises(self):
        with pytest.raises(ValidationError):
            Artifact(name="no_desc")

    def test_multiple_contributors_and_users(self):
        a = Artifact(
            name="shared",
            description="shared artifact",
            contributors=["a", "b", "c"],
            users=["d", "e"],
        )
        assert len(a.contributors) == 3
        assert len(a.users) == 2


# ─────────────────────────────────────────────────────────────
# NODE
# ─────────────────────────────────────────────────────────────

class TestNode:

    def _make_node(self, **kwargs):
        defaults = dict(
            id="test_node",
            name="Test Node",
            node_type=NodeType.DOMAIN,
            system_prompt="You are a test agent.",
        )
        defaults.update(kwargs)
        return Node(**defaults)

    def test_basic_creation(self):
        n = self._make_node()
        assert n.id == "test_node"
        assert n.status == NodeStatus.PENDING
        assert n.query_tool == []
        assert n.mcp_server_path is None

    def test_node_type_enum(self):
        for nt in NodeType:
            n = self._make_node(node_type=nt)
            assert n.node_type == nt

    def test_status_enum(self):
        for ns in NodeStatus:
            n = self._make_node(status=ns)
            assert n.status == ns

    def test_query_tool_typed(self):
        n = self._make_node(
            query_tool=[
                {"tool_description": "evaluate a mathematical expression"},
                {"tool_description": "convert units of measurement"},
            ]
        )
        assert len(n.query_tool) == 2
        assert isinstance(n.query_tool[0], ToolQuery)
        assert n.query_tool[0].tool_description == "evaluate a mathematical expression"

    def test_query_tool_empty_by_default(self):
        n = self._make_node()
        assert n.query_tool == []

    def test_mcp_server_path_can_be_set(self):
        n = self._make_node(mcp_server_path="/some/path/mcp_server.py")
        assert n.mcp_server_path == "/some/path/mcp_server.py"

    def test_model_copy_immutability(self):
        n = self._make_node()
        n2 = n.model_copy(update={"status": NodeStatus.COMPLETED})
        assert n.status == NodeStatus.PENDING
        assert n2.status == NodeStatus.COMPLETED

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            Node(name="x", node_type=NodeType.DOMAIN, system_prompt="y")  # missing id

    def test_contributor_system_prompts(self):
        n = self._make_node(
            node_type=NodeType.SYNTHESIZER,
            contributor_system_prompts=["prompt_a", "prompt_b"],
        )
        assert len(n.contributor_system_prompts) == 2


# ─────────────────────────────────────────────────────────────
# DAG
# ─────────────────────────────────────────────────────────────

class TestDAG:

    def _make_node(self, node_id: str, inputs=None, outputs=None):
        return Node(
            id=node_id,
            name=node_id.replace("_", " ").title(),
            node_type=NodeType.DOMAIN,
            system_prompt=f"You are {node_id}.",
            input_artifacts=inputs or [],
            output_artifacts=outputs or [],
        )

    def _make_artifact(self, name, contributors=None, users=None):
        return Artifact(
            name=name,
            description=f"Artifact {name}",
            contributors=contributors or [],
            users=users or [],
        )

    def test_empty_dag(self):
        dag = DAG(id="run_001", task="test task")
        assert dag.nodes == {}
        assert dag.artifacts == {}
        assert dag.levels == []
        assert dag.status == DAGStatus.PLANNED

    def test_dag_with_valid_nodes_and_artifacts(self):
        node_a = self._make_node("agent_a", outputs=["artifact_1"])
        node_b = self._make_node("agent_b", inputs=["artifact_1"])
        art    = self._make_artifact("artifact_1", contributors=["agent_a"], users=["agent_b"])

        dag = DAG(
            id="run_001",
            task="test",
            nodes={"agent_a": node_a, "agent_b": node_b},
            artifacts={"artifact_1": art},
        )
        assert len(dag.nodes) == 2
        assert len(dag.artifacts) == 1

    def test_dag_validator_rejects_unknown_contributor(self):
        art = self._make_artifact("art_1", contributors=["ghost_node"], users=[])
        with pytest.raises(ValueError, match="ghost_node"):
            DAG(id="run_001", task="t", artifacts={"art_1": art})

    def test_dag_validator_rejects_unknown_user(self):
        node_a = self._make_node("agent_a")
        art = self._make_artifact("art_1", contributors=["agent_a"], users=["ghost_node"])
        with pytest.raises(ValueError, match="ghost_node"):
            DAG(
                id="run_001",
                task="t",
                nodes={"agent_a": node_a},
                artifacts={"art_1": art},
            )

    def test_get_node(self):
        node_a = self._make_node("agent_a")
        dag = DAG(id="r", task="t", nodes={"agent_a": node_a})
        assert dag.get_node("agent_a") is node_a

    def test_get_node_missing_raises(self):
        dag = DAG(id="r", task="t")
        with pytest.raises(KeyError):
            dag.get_node("does_not_exist")

    def test_get_artifact(self):
        node_a = self._make_node("agent_a")
        art = self._make_artifact("art_1", contributors=["agent_a"])
        dag = DAG(
            id="r", task="t",
            nodes={"agent_a": node_a},
            artifacts={"art_1": art},
        )
        assert dag.get_artifact("art_1") is art

    def test_is_complete_false_when_pending(self):
        node_a = self._make_node("agent_a")
        dag = DAG(id="r", task="t", nodes={"agent_a": node_a})
        assert not dag.is_complete()

    def test_is_complete_true_when_all_completed(self):
        node_a = self._make_node("agent_a")
        node_a = node_a.model_copy(update={"status": NodeStatus.COMPLETED})
        dag = DAG(id="r", task="t", nodes={"agent_a": node_a})
        assert dag.is_complete()

    def test_dag_status_transitions(self):
        dag = DAG(id="r", task="t")
        assert dag.status == DAGStatus.PLANNED
        dag2 = dag.model_copy(update={"status": DAGStatus.RUNNING})
        assert dag2.status == DAGStatus.RUNNING
        dag3 = dag2.model_copy(update={"status": DAGStatus.COMPLETED})
        assert dag3.status == DAGStatus.COMPLETED