"""
tests/conftest.py

Shared pytest fixtures for Coven test suite.
"""

import pytest
from coven.models import Artifact, Node, NodeType, NodeStatus, DAG


@pytest.fixture
def sample_artifact():
    return Artifact(
        name="sample_artifact",
        description="A sample artifact for testing.",
        contributors=["agent_a"],
        users=["agent_b"],
        body={"key": "value"},
    )


@pytest.fixture
def sample_node():
    return Node(
        id="sample_agent",
        name="Sample Agent",
        node_type=NodeType.DOMAIN,
        system_prompt="You are a sample agent for testing.",
        input_artifacts=[],
        output_artifacts=["sample_artifact"],
    )


@pytest.fixture
def two_node_dag():
    """
    Minimal valid DAG:
        agent_a → artifact_1 → agent_b
    """
    node_a = Node(
        id="agent_a",
        name="Agent A",
        node_type=NodeType.DOMAIN,
        system_prompt="You are agent A.",
        output_artifacts=["artifact_1"],
    )
    node_b = Node(
        id="agent_b",
        name="Agent B",
        node_type=NodeType.DOMAIN,
        system_prompt="You are agent B.",
        input_artifacts=["artifact_1"],
        output_artifacts=["artifact_2"],
    )
    artifact_1 = Artifact(
        name="artifact_1",
        description="Output from agent A.",
        contributors=["agent_a"],
        users=["agent_b"],
    )
    artifact_2 = Artifact(
        name="artifact_2",
        description="Output from agent B.",
        contributors=["agent_b"],
        users=[],
    )
    return DAG(
        id="test_dag",
        task="Test two-node pipeline",
        nodes={"agent_a": node_a, "agent_b": node_b},
        artifacts={"artifact_1": artifact_1, "artifact_2": artifact_2},
        levels=[["agent_a"], ["agent_b"]],
    )