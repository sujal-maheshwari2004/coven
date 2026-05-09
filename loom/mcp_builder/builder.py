from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from loom.models import Node

logger = logging.getLogger(__name__)


class MCPNodeBuilder:
    """
    Wraps ToolStorePy to build a dedicated MCP server for a single DAG node.

    Called by AgentRunner before executing any domain node that has
    non-empty query_tool entries.

    Workflow per node:
        1. Write node's query_tool list as queries.json
        2. Call ToolStorePy.build() pointing at that queries.json
        3. Return path to generated mcp_unified_server.py
        4. AgentRunner stores path on node.mcp_server_path

    Each node gets its own isolated workspace under the run's base workspace,
    so parallel nodes in the same level never conflict.
    """

    def __init__(
        self,
        base_workspace: Path,
        index: str | None = "core-tools",
        index_url: str | None = None,
        install_requirements: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            base_workspace: Root workspace for this DAG run.
                            Node workspaces will be created as subdirectories.
            index: ToolStorePy built-in index name (default: "core-tools").
            index_url: Direct URL to a custom index archive. Overrides index.
            install_requirements: Whether to install repo requirements in venv.
            verbose: Enable verbose ToolStorePy logging.
        """
        self.base_workspace      = Path(base_workspace)
        self.index               = index
        self.index_url           = index_url
        self.install_requirements = install_requirements
        self.verbose             = verbose

        self.base_workspace.mkdir(parents=True, exist_ok=True)

    def build_for_node(self, node: Node) -> Path | None:
        """
        Build an MCP server for a node's tool requirements.

        Args:
            node: The Node whose query_tool list defines what tools to build.

        Returns:
            Path to mcp_unified_server.py, or None if node has no tool queries.

        Raises:
            RuntimeError: If ToolStorePy build fails.
        """
        if not node.query_tool:
            return None

        from toolstorepy import ToolStorePy  # imported here so loom works without it if unused

        node_workspace = self.base_workspace / "mcp" / node.id
        node_workspace.mkdir(parents=True, exist_ok=True)

        queries_path = self._write_queries(node, node_workspace)

        logger.info(
            f"[MCPNodeBuilder] Building MCP server for node '{node.id}' "
            f"with {len(node.query_tool)} tool queries..."
        )

        try:
            toolstore = ToolStorePy(
                workspace=str(node_workspace),
                install_requirements=self.install_requirements,
                verbose=self.verbose,
            )

            output_path = toolstore.build(
                queries=str(queries_path),
                index=self.index if not self.index_url else None,
                index_url=self.index_url,
                force_refresh=False,
            )

            logger.info(
                f"[MCPNodeBuilder] MCP server built for node '{node.id}' → {output_path}"
            )

            return Path(output_path)

        except Exception as e:
            raise RuntimeError(
                f"ToolStorePy failed to build MCP server for node '{node.id}': {e}"
            ) from e

    def _write_queries(self, node: Node, workspace: Path) -> Path:
        """
        Write node's query_tool list as a ToolStorePy-compatible queries.json.

        Format:
            [{"tool_description": "..."}]

        Args:
            node: Node whose query_tool to serialize.
            workspace: Directory to write queries.json into.

        Returns:
            Path to written queries.json.
        """
        queries = [{"tool_description": tq.tool_description} for tq in node.query_tool]
        queries_path = workspace / "queries.json"

        with open(queries_path, "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=2)

        logger.debug(
            f"[MCPNodeBuilder] Wrote {len(queries)} queries for node '{node.id}' "
            f"→ {queries_path}"
        )

        return queries_path