from __future__ import annotations

import asyncio
import logging

from loom.models import DAG, Node, NodeStatus, DAGStatus
from .artifact_store import ArtifactStore
from .agent_runner import AgentRunner


logger = logging.getLogger(__name__)


class Executor:
    """
    Stage 4 — Level-wise async DAG execution.

    Iterates through topological levels in order. Within each level,
    all nodes are launched concurrently via asyncio.gather().

    After each level completes:
    - Node statuses are updated in the DAG
    - Artifact store is populated with produced bodies
    - Any failed nodes are reported and execution halts

    This guarantees that when level N begins, all artifacts
    from levels 0..N-1 are fully available in the store.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model   = model
        self._runner = AgentRunner(model=model)

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

        dag = dag.model_copy(update={"status": DAGStatus.RUNNING})

        for level_index, level in enumerate(dag.levels):
            logger.info(
                f"Executing level {level_index}/{len(dag.levels) - 1}: "
                f"{level}"
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
            "status": DAGStatus.COMPLETED,
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

        Args:
            level: List of node IDs to execute in parallel.
            dag: Current DAG state.
            store: Shared artifact store.

        Returns:
            Updated (DAG, store) after level completes.
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

        # ── Update node statuses in DAG ───────────────────────────────────────
        new_nodes = dict(dag.nodes)
        for node in updated_nodes:
            new_nodes[node.id] = node
            logger.info(f"  Node '{node.id}' → {node.status.value}")

        dag = dag.model_copy(update={"nodes": new_nodes})

        return dag, store