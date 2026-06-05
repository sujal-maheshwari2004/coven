from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from coven.models import DAG, Node, NodeStatus, DAGStatus
from coven.mcp_builder import MCPNodeBuilder
from .artifact_store import ArtifactStore
from .agent_runner import AgentRunner


logger = logging.getLogger(__name__)


class Executor:
    """
    Stage 4 — Level-wise async DAG execution.

    Iterates through topological levels in order. Within each level,
    all nodes are launched concurrently via asyncio.gather().

    If a MCPNodeBuilder is provided, domain nodes with non-empty
    query_tool will have their MCP server built before execution.
    MCP builds within a level also run in parallel.

    After each level completes:
    - Node statuses are updated in the DAG
    - Artifact store is populated with produced bodies
    - Any failed nodes halt execution
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        mcp_builder: MCPNodeBuilder | None = None,
    ):
        self.model   = model
        self._runner = AgentRunner(model=model, mcp_builder=mcp_builder)

    async def execute(self, dag: DAG) -> DAG:
        """
        Execute the full DAG level by level.

        Args:
            dag: Fully planned DAG with levels populated by the sorter
                 and synthesizer nodes injected.

        Returns:
            Updated DAG with all node statuses and artifact bodies populated.
        """
        store = ArtifactStore(dag.artifacts)
        dag   = dag.model_copy(update={"status": DAGStatus.RUNNING})

        for level_index, level in enumerate(dag.levels):
            logger.info(
                f"Executing level {level_index}/{len(dag.levels) - 1}: {level}"
            )

            dag, store = await self._execute_level(
                level=level,
                dag=dag,
                store=store,
            )

            failed = [
                node_id for node_id in level
                if dag.nodes[node_id].status == NodeStatus.FAILED
            ]

            if failed:
                logger.error(f"Level {level_index} had failures: {failed}")
                dag = dag.model_copy(update={"status": DAGStatus.FAILED})
                return dag

        # ── Write final artifact bodies back into DAG ─────────────────────────
        final_artifacts = await store.all()
        dag = dag.model_copy(update={
            "artifacts": final_artifacts,
            "status":    DAGStatus.COMPLETED,
        })

        logger.info("DAG execution completed successfully.")
        return dag

    async def _execute_level(
        self,
        level: list[str],
        dag: DAG,
        store: ArtifactStore,
    ) -> tuple[DAG, ArtifactStore]:
        """
        Execute all nodes in a single level concurrently.

        MCP builds and agent LLM calls all run inside asyncio.gather —
        ToolStorePy's blocking build() is wrapped in run_in_executor
        inside MCPNodeBuilder so it doesn't block the event loop.
        """
        tasks = [
            self._runner.run(
                node=dag.nodes[node_id],
                store=store,
                nodes=dag.nodes,
            )
            for node_id in level
        ]

        updated_nodes: list[Node] = await asyncio.gather(*tasks)

        new_nodes = dict(dag.nodes)
        for node in updated_nodes:
            new_nodes[node.id] = node
            logger.info(f"  Node '{node.id}' → {node.status.value}")

        dag = dag.model_copy(update={"nodes": new_nodes})
        return dag, store