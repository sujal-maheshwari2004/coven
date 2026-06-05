"""
tests/mcp_builder/test_mcp_builder.py

Unit tests for MCPNodeBuilder.
ToolStorePy.build() is mocked — no network calls or real builds.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from coven.mcp_builder.builder import MCPNodeBuilder
from coven.models import Node, NodeType, NodeStatus
from coven.models.node import ToolQuery


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _make_node(id, query_tool=None):
    return Node(
        id=id,
        name=id.title(),
        node_type=NodeType.DOMAIN,
        system_prompt=f"You are {id}.",
        query_tool=query_tool or [],
        output_artifacts=[f"{id}_out"],
    )


def _make_builder(tmp_path):
    return MCPNodeBuilder(
        base_workspace=tmp_path,
        index="core-tools",
        index_url=None,
        install_requirements=False,
        verbose=False,
    )


# ─────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────

class TestMCPNodeBuilder:

    def test_build_for_node_returns_none_when_no_query_tool(self, tmp_path):
        builder = _make_builder(tmp_path)
        node = _make_node("agent_a", query_tool=[])
        result = builder.build_for_node(node)
        assert result is None

    def test_build_for_node_returns_path_when_toolstorepy_succeeds(self, tmp_path):
        builder = _make_builder(tmp_path)
        node = _make_node("agent_a", query_tool=[
            ToolQuery(tool_description="evaluate a mathematical expression"),
        ])

        mock_output = tmp_path / "mcp" / "agent_a" / "mcp_unified_server.py"
        mock_output.parent.mkdir(parents=True, exist_ok=True)
        mock_output.touch()

        mock_toolstore = MagicMock()
        mock_toolstore.build.return_value = str(mock_output)

        with patch("coven.mcp_builder.builder.ToolStorePy", return_value=mock_toolstore):
            result = builder.build_for_node(node)

        assert result is not None
        assert result.path == mock_output
        assert isinstance(result.port, int)

    def test_build_for_node_raises_on_toolstorepy_failure(self, tmp_path):
        builder = _make_builder(tmp_path)
        node = _make_node("agent_a", query_tool=[
            ToolQuery(tool_description="some tool"),
        ])

        mock_toolstore = MagicMock()
        mock_toolstore.build.side_effect = RuntimeError("Build failed")

        with patch("coven.mcp_builder.builder.ToolStorePy", return_value=mock_toolstore):
            with pytest.raises(RuntimeError, match="ToolStorePy failed"):
                builder.build_for_node(node)

    def test_write_queries_creates_file(self, tmp_path):
        builder = _make_builder(tmp_path)
        node = _make_node("agent_a", query_tool=[
            ToolQuery(tool_description="evaluate a mathematical expression"),
            ToolQuery(tool_description="convert units of measurement"),
        ])
        workspace = tmp_path / "test_workspace"
        workspace.mkdir()

        queries_path = builder._write_queries(node, workspace)

        assert queries_path.exists()
        assert queries_path.name == "queries.json"

    def test_write_queries_correct_format(self, tmp_path):
        builder = _make_builder(tmp_path)
        node = _make_node("agent_a", query_tool=[
            ToolQuery(tool_description="evaluate a mathematical expression"),
            ToolQuery(tool_description="convert units of measurement"),
        ])
        workspace = tmp_path / "ws"
        workspace.mkdir()

        queries_path = builder._write_queries(node, workspace)

        with open(queries_path) as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0] == {"tool_description": "evaluate a mathematical expression"}
        assert data[1] == {"tool_description": "convert units of measurement"}

    def test_each_node_gets_isolated_workspace(self, tmp_path):
        builder = _make_builder(tmp_path)

        node_a = _make_node("agent_a", query_tool=[ToolQuery(tool_description="tool a")])
        node_b = _make_node("agent_b", query_tool=[ToolQuery(tool_description="tool b")])

        mock_toolstore = MagicMock()

        def fake_build(queries, index, index_url, force_refresh):
            # Return a path based on the workspace being built
            ws = Path(queries).parent
            out = ws / "mcp_unified_server.py"
            out.touch()
            return str(out)

        mock_toolstore.build.side_effect = fake_build

        with patch("coven.mcp_builder.builder.ToolStorePy", return_value=mock_toolstore):
            path_a = builder.build_for_node(node_a)
            path_b = builder.build_for_node(node_b)

        assert path_a != path_b
        assert "agent_a" in str(path_a)
        assert "agent_b" in str(path_b)

    def test_toolstorepy_called_with_correct_index(self, tmp_path):
        builder = MCPNodeBuilder(
            base_workspace=tmp_path,
            index="core-tools",
            index_url=None,
        )
        node = _make_node("agent_a", query_tool=[ToolQuery(tool_description="some tool")])

        mock_toolstore = MagicMock()
        mock_toolstore.build.return_value = str(tmp_path / "mcp_unified_server.py")

        with patch("coven.mcp_builder.builder.ToolStorePy", return_value=mock_toolstore) as MockTS:
            builder.build_for_node(node)

        call_kwargs = mock_toolstore.build.call_args
        assert call_kwargs.kwargs.get("index") == "core-tools" or \
               (call_kwargs.args and "core-tools" in str(call_kwargs))

    def test_toolstorepy_called_with_index_url_when_provided(self, tmp_path):
        builder = MCPNodeBuilder(
            base_workspace=tmp_path,
            index=None,
            index_url="https://example.com/my-index.zip",
        )
        node = _make_node("agent_a", query_tool=[ToolQuery(tool_description="some tool")])

        mock_toolstore = MagicMock()
        mock_toolstore.build.return_value = str(tmp_path / "mcp_unified_server.py")

        with patch("coven.mcp_builder.builder.ToolStorePy", return_value=mock_toolstore):
            builder.build_for_node(node)

        call_kwargs = mock_toolstore.build.call_args
        assert call_kwargs.kwargs.get("index_url") == "https://example.com/my-index.zip" or \
               "https://example.com/my-index.zip" in str(call_kwargs)