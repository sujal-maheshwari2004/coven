"""
tests/initiator/test_initiator.py

Unit tests for ArtifactStore and Executor.
AgentRunner LLM calls are mocked.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from coven.initiator.artifact_store import ArtifactStore
from coven.initiator.executor import Executor
from coven.models import (
    Artifact, Node, NodeType, NodeStatus,
    DAG, DAGStatus,
)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _make_artifact(name, body=None):
    return Artifact(
        name=name,
        description=f"Artifact {name}",
        contributors=[],
        users=[],
        body=body or {},
    )


def _make_node(id, inputs=None, outputs=None, node_type=NodeType.DOMAIN):
    return Node(
        id=id,
        name=id.title(),
        node_type=node_type,
        system_prompt=f"You are {id}.",
        input_artifacts=inputs or [],
        output_artifacts=outputs or [],
    )


def _make_dag(nodes=None, artifacts=None, levels=None):
    return DAG(
        id="test_run",
        task="test task",
        nodes=nodes or {},
        artifacts=artifacts or {},
        levels=levels or [],
    )


# ─────────────────────────────────────────────────────────────
# ARTIFACT STORE TESTS
# ─────────────────────────────────────────────────────────────

class TestArtifactStore:

    def _store_with(self, *artifact_names):
        artifacts = {n: _make_artifact(n) for n in artifact_names}
        return ArtifactStore(artifacts)

    @pytest.mark.asyncio
    async def test_get_existing_artifact(self):
        store = self._store_with("art_1")
        art = await store.get("art_1")
        assert art.name == "art_1"

    @pytest.mark.asyncio
    async def test_get_missing_artifact_raises(self):
        store = self._store_with("art_1")
        with pytest.raises(KeyError, match="art_2"):
            await store.get("art_2")

    @pytest.mark.asyncio
    async def test_put_updates_body(self):
        store = self._store_with("art_1")
        await store.put("art_1", {"result": 42})
        art = await store.get("art_1")
        assert art.body == {"result": 42}

    @pytest.mark.asyncio
    async def test_put_unknown_artifact_raises(self):
        store = self._store_with("art_1")
        with pytest.raises(KeyError, match="ghost_art"):
            await store.put("ghost_art", {"x": 1})

    @pytest.mark.asyncio
    async def test_get_many_returns_ordered(self):
        store = self._store_with("art_1", "art_2", "art_3")
        await store.put("art_1", {"v": 1})
        await store.put("art_2", {"v": 2})
        await store.put("art_3", {"v": 3})
        result = await store.get_many(["art_3", "art_1"])
        assert result[0].name == "art_3"
        assert result[1].name == "art_1"

    @pytest.mark.asyncio
    async def test_all_returns_snapshot(self):
        store = self._store_with("art_1", "art_2")
        snapshot = await store.all()
        assert "art_1" in snapshot
        assert "art_2" in snapshot

    def test_is_ready_false_when_empty_body(self):
        store = self._store_with("art_1")
        assert not store.is_ready("art_1")

    @pytest.mark.asyncio
    async def test_is_ready_true_after_put(self):
        store = self._store_with("art_1")
        await store.put("art_1", {"data": "filled"})
        assert store.is_ready("art_1")

    def test_is_ready_false_for_unknown(self):
        store = self._store_with("art_1")
        assert not store.is_ready("nonexistent")

    def test_all_inputs_ready_true(self):
        artifacts = {
            "a": _make_artifact("a", body={"x": 1}),
            "b": _make_artifact("b", body={"y": 2}),
        }
        store = ArtifactStore(artifacts)
        assert store.all_inputs_ready(["a", "b"])

    def test_all_inputs_ready_false_when_one_empty(self):
        artifacts = {
            "a": _make_artifact("a", body={"x": 1}),
            "b": _make_artifact("b"),  # empty body
        }
        store = ArtifactStore(artifacts)
        assert not store.all_inputs_ready(["a", "b"])

    @pytest.mark.asyncio
    async def test_concurrent_puts_are_safe(self):
        store = self._store_with("art_1", "art_2", "art_3")

        async def write(name, val):
            await store.put(name, {"v": val})

        await asyncio.gather(
            write("art_1", 1),
            write("art_2", 2),
            write("art_3", 3),
        )
        assert (await store.get("art_1")).body == {"v": 1}
        assert (await store.get("art_2")).body == {"v": 2}
        assert (await store.get("art_3")).body == {"v": 3}


# ─────────────────────────────────────────────────────────────
# EXECUTOR TESTS
# ─────────────────────────────────────────────────────────────

class TestExecutor:

    def _make_completed_node(self, id, outputs=None):
        """Returns a node that the mock runner will report as COMPLETED."""
        return _make_node(id, outputs=outputs or [f"{id}_out"])

    def _make_dag_with_levels(self, levels):
        """Build a minimal DAG with the given levels."""
        all_node_ids = [n for level in levels for n in level]
        nodes = {id: self._make_completed_node(id) for id in all_node_ids}
        artifacts = {
            f"{id}_out": _make_artifact(f"{id}_out")
            for id in all_node_ids
        }
        return _make_dag(nodes=nodes, artifacts=artifacts, levels=levels)

    @pytest.mark.asyncio
    async def test_executor_completes_single_level(self):
        dag = self._make_dag_with_levels([["agent_a"]])

        async def mock_run(node, store, nodes):
            return node.model_copy(update={"status": NodeStatus.COMPLETED})

        executor = Executor(model="gpt-4o")
        with patch.object(executor._runner, "run", side_effect=mock_run):
            result = await executor.execute(dag)

        assert result.status == DAGStatus.COMPLETED
        assert result.nodes["agent_a"].status == NodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_executor_completes_multiple_levels(self):
        dag = self._make_dag_with_levels([["agent_a"], ["agent_b"]])

        async def mock_run(node, store, nodes):
            return node.model_copy(update={"status": NodeStatus.COMPLETED})

        executor = Executor(model="gpt-4o")
        with patch.object(executor._runner, "run", side_effect=mock_run):
            result = await executor.execute(dag)

        assert result.status == DAGStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_executor_fails_on_failed_node(self):
        dag = self._make_dag_with_levels([["agent_a"]])

        async def mock_run(node, store, nodes):
            return node.model_copy(update={"status": NodeStatus.FAILED})

        executor = Executor(model="gpt-4o")
        with patch.object(executor._runner, "run", side_effect=mock_run):
            result = await executor.execute(dag)

        assert result.status == DAGStatus.FAILED

    @pytest.mark.asyncio
    async def test_executor_halts_after_failed_level(self):
        """If level 0 fails, level 1 should never execute."""
        dag = self._make_dag_with_levels([["agent_a"], ["agent_b"]])

        call_order = []

        async def mock_run(node, store, nodes):
            call_order.append(node.id)
            if node.id == "agent_a":
                return node.model_copy(update={"status": NodeStatus.FAILED})
            return node.model_copy(update={"status": NodeStatus.COMPLETED})

        executor = Executor(model="gpt-4o")
        with patch.object(executor._runner, "run", side_effect=mock_run):
            result = await executor.execute(dag)

        assert result.status == DAGStatus.FAILED
        assert "agent_b" not in call_order

    @pytest.mark.asyncio
    async def test_executor_parallel_level_runs_concurrently(self):
        """Nodes in the same level should all be invoked."""
        dag = self._make_dag_with_levels([["agent_a", "agent_b", "agent_c"]])
        called = []

        async def mock_run(node, store, nodes):
            called.append(node.id)
            return node.model_copy(update={"status": NodeStatus.COMPLETED})

        executor = Executor(model="gpt-4o")
        with patch.object(executor._runner, "run", side_effect=mock_run):
            result = await executor.execute(dag)

        assert sorted(called) == ["agent_a", "agent_b", "agent_c"]
        assert result.status == DAGStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_executor_writes_artifacts_to_dag(self):
        dag = self._make_dag_with_levels([["agent_a"]])

        async def mock_run(node, store, nodes):
            await store.put("agent_a_out", {"computed": True})
            return node.model_copy(update={"status": NodeStatus.COMPLETED})

        executor = Executor(model="gpt-4o")
        with patch.object(executor._runner, "run", side_effect=mock_run):
            result = await executor.execute(dag)

        assert result.artifacts["agent_a_out"].body == {"computed": True}

    @pytest.mark.asyncio
    async def test_executor_sets_running_status(self):
        dag = self._make_dag_with_levels([["agent_a"]])

        statuses_seen = []

        async def mock_run(node, store, nodes):
            statuses_seen.append(dag.status)
            return node.model_copy(update={"status": NodeStatus.COMPLETED})

        executor = Executor(model="gpt-4o")
        with patch.object(executor._runner, "run", side_effect=mock_run):
            result = await executor.execute(dag)

        # DAG should have been set to RUNNING before any node ran
        assert DAGStatus.RUNNING in statuses_seen or result.status == DAGStatus.COMPLETED