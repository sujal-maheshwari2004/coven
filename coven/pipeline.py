from __future__ import annotations

import logging
import uuid
from pathlib import Path

from coven.models import DAG, DAGStatus
from coven.decomposer import DecomposerAgent, DecomposerParser
from coven.graph_builder import GraphBuilderAgent, GraphBuilderParser, GraphBuilderValidator
from coven.sorter import TopologicalSorter, SorterValidator
from coven.synthesizer import SynthesizerInjector
from coven.mcp_builder import MCPNodeBuilder
from coven.initiator import Executor
from coven.compiler import CompilerAgent, CompilerFormatter, CompilerResponse, OutputSection


logger = logging.getLogger(__name__)


class Coven:
    """
    Top-level orchestrator for the Coven pipeline.

    Runs all five stages in sequence:
        1. Decomposer     — breaks task into nodes + artifacts
        2. Graph Builder  — validates and formalizes DAG edges
        3. Sorter         — topological sort into execution levels
        4. Initiator      — level-wise async agent execution
                            (with per-node MCP server builds via ToolStorePy)
        5. Compiler       — assembles final output from all artifacts

    Usage:
        coven = Coven(model="gpt-4o")
        dag  = await coven.run("Produce a go-to-market strategy for a B2B SaaS product")
        print(coven.to_text(dag))

    Tool usage:
        Nodes whose query_tool list is non-empty will have a ToolStorePy MCP
        server built for them automatically before execution. The decomposer
        LLM decides which tools each node needs — just describe the task and
        Coven handles the rest.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        workspace: str | Path = "coven_workspace",
        mcp_index: str | None = "core-tools",
        mcp_index_url: str | None = None,
        mcp_install_requirements: bool = False,
        mcp_host: str = "0.0.0.0",
        mcp_base_port: int = 8100,
        mcp_llm_scan: bool = False,
        mcp_llm_model: str = "claude-sonnet-4-6",
        mcp_verbose: bool = False,
    ):
        """
        Args:
            model: LiteLLM-compatible model string (e.g. "gpt-4o", "claude-sonnet-4-6").
            workspace: Root directory for all run artifacts and MCP workspaces.
            mcp_index: ToolStorePy built-in index name. Default: "core-tools".
            mcp_index_url: Direct URL to a custom ToolStorePy index. Overrides mcp_index.
            mcp_install_requirements: Install repo requirements in MCP venv.
            mcp_host: Host MCP servers bind on. Default: "0.0.0.0".
            mcp_base_port: Starting port for MCP servers; each node gets the next port.
            mcp_llm_scan: Use an LLM to review each tool repo autonomously (no human prompt).
            mcp_llm_model: Model for LLM security scanning. Any LiteLLM/LangChain string.
            mcp_verbose: Enable verbose ToolStorePy logging.
        """
        self.model     = model
        self.workspace = Path(workspace)

        # ── MCP builder — shared across all nodes in a run ────────────────────
        self._mcp_builder = MCPNodeBuilder(
            base_workspace=self.workspace,
            index=mcp_index if not mcp_index_url else None,
            index_url=mcp_index_url,
            install_requirements=mcp_install_requirements,
            host=mcp_host,
            base_port=mcp_base_port,
            llm_scan=mcp_llm_scan,
            llm_model=mcp_llm_model,
            verbose=mcp_verbose,
        )

        # ── Pipeline stages ───────────────────────────────────────────────────
        self._decomposer        = DecomposerAgent(model=model)
        self._decomposer_parser = DecomposerParser()
        self._graph_builder     = GraphBuilderAgent(model=model)
        self._graph_parser      = GraphBuilderParser()
        self._graph_validator   = GraphBuilderValidator()
        self._sorter            = TopologicalSorter()
        self._sorter_validator  = SorterValidator()
        self._synth_injector    = SynthesizerInjector()
        self._executor          = Executor(model=model, mcp_builder=self._mcp_builder)
        self._compiler          = CompilerAgent(model=model)
        self._formatter         = CompilerFormatter()

    async def run(self, task: str) -> DAG:
        """
        Execute the full Coven pipeline for a given task.

        Args:
            task: The complex task to solve.

        Returns:
            Completed DAG with final_output populated.
        """
        dag_id = str(uuid.uuid4())[:8]
        logger.info(f"[{dag_id}] Starting Coven pipeline: {task[:80]}...")

        # ── Stage 1: Decompose ────────────────────────────────────────────────
        logger.info(f"[{dag_id}] Stage 1: Decomposing task...")
        decomp_response      = await self._decomposer.arun(task)
        nodes, artifacts     = self._decomposer_parser.parse(decomp_response)
        logger.info(f"[{dag_id}] → {len(nodes)} nodes, {len(artifacts)} artifacts.")

        # ── Stage 2: Build Graph ──────────────────────────────────────────────
        logger.info(f"[{dag_id}] Stage 2: Building graph...")
        graph_response       = await self._graph_builder.arun(nodes, artifacts)
        nodes, artifacts, edges, synthesizer_targets = self._graph_parser.parse(graph_response)
        G                    = self._graph_validator.validate(nodes, artifacts, edges)
        logger.info(f"[{dag_id}] → {len(edges)} edges, synthesizer targets: {synthesizer_targets}")

        # ── Stage 2b: Inject Synthesizer Nodes ───────────────────────────────
        if synthesizer_targets:
            logger.info(f"[{dag_id}] Injecting synthesizer nodes...")
            nodes, artifacts, G = self._synth_injector.inject(
                nodes, artifacts, synthesizer_targets, G
            )
            logger.info(f"[{dag_id}] → {len(nodes)} total nodes after injection.")

        # ── Stage 3: Topological Sort ─────────────────────────────────────────
        logger.info(f"[{dag_id}] Stage 3: Sorting DAG...")
        self._sorter_validator.validate(G, nodes, artifacts)
        levels = self._sorter.sort(G)
        logger.info(f"[{dag_id}]\n{self._sorter.describe(G, nodes)}")

        # ── Assemble DAG ──────────────────────────────────────────────────────
        dag = DAG(
            id=dag_id,
            task=task,
            nodes=nodes,
            artifacts=artifacts,
            levels=levels,
        )

        # ── Stage 4: Execute ──────────────────────────────────────────────────
        logger.info(f"[{dag_id}] Stage 4: Executing {len(levels)} levels...")
        dag = await self._executor.execute(dag)

        if dag.status == DAGStatus.FAILED:
            logger.error(f"[{dag_id}] Pipeline execution failed.")
            return dag

        # ── Stage 5: Compile ──────────────────────────────────────────────────
        logger.info(f"[{dag_id}] Stage 5: Compiling final output...")
        compiler_response = await self._compiler.arun(dag)
        dag               = self._formatter.format(compiler_response, dag)

        logger.info(f"[{dag_id}] Pipeline complete.")
        return dag

    def to_text(self, dag: DAG) -> str:
        """Render the final DAG output as plain text."""
        if not dag.final_output:
            return "Pipeline did not produce a final output."

        response = CompilerResponse(
            title=dag.final_output.get("title", ""),
            summary=dag.final_output.get("summary", ""),
            sections=[
                OutputSection(**s) for s in dag.final_output.get("sections", [])
            ],
            recommendations=dag.final_output.get("recommendations", []),
            metadata=dag.final_output.get("metadata", {}),
        )
        return self._formatter.to_text(response)