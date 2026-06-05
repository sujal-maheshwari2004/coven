from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from coven.models import Node

# Import at module level so tests can patch "coven.mcp_builder.builder.ToolStorePy"
try:
    from toolstorepy import ToolStorePy
except ImportError:  # allow Coven to load even if toolstorepy isn't installed
    ToolStorePy = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


@dataclass
class MCPBuildResult:
    path: Path
    port: int


class MCPNodeBuilder:
    """
    Wraps ToolStorePy to build a dedicated MCP server for a single DAG node.

    Called by AgentRunner before executing any domain node that has
    non-empty query_tool entries.

    Each node gets its own isolated workspace and port so parallel nodes
    in the same execution level never conflict.
    """

    def __init__(
        self,
        base_workspace: Path,
        index: str | None = "core-tools",
        index_url: str | None = None,
        install_requirements: bool = False,
        host: str = "0.0.0.0",
        base_port: int = 8100,
        llm_scan: bool = False,
        llm_model: str = "claude-sonnet-4-6",
        verbose: bool = False,
    ):
        self.base_workspace       = Path(base_workspace)
        self.index                = index
        self.index_url            = index_url
        self.install_requirements = install_requirements
        self.host                 = host
        self.llm_scan             = llm_scan
        self.llm_model            = llm_model
        self.verbose              = verbose

        self._port_counter = itertools.count(base_port)

        self.base_workspace.mkdir(parents=True, exist_ok=True)

    def build_for_node(self, node: Node) -> MCPBuildResult | None:
        """
        Build an MCP server for a node's tool requirements.

        Returns:
            MCPBuildResult with path and port, or None if node has no tool queries.

        Raises:
            RuntimeError: If ToolStorePy is not installed or build fails.
        """
        if not node.query_tool:
            return None

        if ToolStorePy is None:
            raise RuntimeError(
                "toolstorepy is not installed. Run: pip install toolstorepy"
            )

        node_workspace = self.base_workspace / "mcp" / node.id
        node_workspace.mkdir(parents=True, exist_ok=True)

        queries_path = self._write_queries(node, node_workspace)
        port = next(self._port_counter)

        logger.info(
            f"[MCPNodeBuilder] Building MCP server for node '{node.id}' "
            f"({len(node.query_tool)} tool queries, port={port})..."
        )

        try:
            toolstore = ToolStorePy(
                workspace=str(node_workspace),
                install_requirements=self.install_requirements,
                host=self.host,
                port=port,
                llm_scan=self.llm_scan,
                llm_model=self.llm_model,
                verbose=self.verbose,
            )

            output_path = toolstore.build(
                queries=str(queries_path),
                index=self.index if not self.index_url else None,
                index_url=self.index_url,
                force_refresh=False,
            )

            logger.info(
                f"[MCPNodeBuilder] MCP server built for node '{node.id}' "
                f"→ {output_path} (port={port})"
            )

            return MCPBuildResult(path=Path(output_path), port=port)

        except Exception as e:
            raise RuntimeError(
                f"ToolStorePy failed to build MCP server for node '{node.id}': {e}"
            ) from e

    def _write_queries(self, node: Node, workspace: Path) -> Path:
        queries = [{"tool_description": tq.tool_description} for tq in node.query_tool]
        queries_path = workspace / "queries.json"

        with open(queries_path, "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=2)

        logger.debug(
            f"[MCPNodeBuilder] Wrote {len(queries)} queries for node '{node.id}' "
            f"→ {queries_path}"
        )

        return queries_path
