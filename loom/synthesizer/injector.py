from __future__ import annotations

from pathlib import Path

import networkx as nx

from loom.models import Artifact, Node, NodeType, NodeStatus


_SYNTH_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "synthesizer.txt"


def _load_synth_prompt() -> str:
    return _SYNTH_PROMPT_PATH.read_text(encoding="utf-8")


class SynthesizerInjector:
    """
    Scans the DAG for artifacts with multiple contributors and
    auto-injects a Synthesizer node between the contributors and
    the artifact's downstream users.

    Injection pattern — before:
        [Agent A] ──┐
                    ├──→ artifact ──→ [User C]
        [Agent B] ──┘

    After injection:
        [Agent A] ──┐
                    ├──→ partial_artifact_a/b ──→ [Synthesizer] ──→ artifact ──→ [User C]
        [Agent B] ──┘

    The synthesizer node:
    - Receives all partial contributions as input artifacts
    - Produces the final merged artifact as its output
    - Carries the system prompts of all contributors for context
    - Gets its own synthesizer system prompt

    The original artifact's contributors list is replaced with
    just the synthesizer node ID, keeping downstream users unchanged.
    """

    def inject(
        self,
        nodes: dict[str, Node],
        artifacts: dict[str, Artifact],
        synthesizer_targets: list[str],
        G: nx.DiGraph,
    ) -> tuple[dict[str, Node], dict[str, Artifact], nx.DiGraph]:
        """
        Inject synthesizer nodes for all targeted artifacts.

        Args:
            nodes: Current node registry.
            artifacts: Current artifact registry.
            synthesizer_targets: Artifact names flagged for synthesis.
            G: Current DAG graph (will be mutated).

        Returns:
            Updated (nodes, artifacts, G) with synthesizer nodes injected.
        """
        synth_prompt = _load_synth_prompt()

        for artifact_name in synthesizer_targets:
            if artifact_name not in artifacts:
                continue

            artifact = artifacts[artifact_name]

            if len(artifact.contributors) <= 1:
                continue

            nodes, artifacts, G = self._inject_for_artifact(
                artifact_name=artifact_name,
                artifact=artifact,
                nodes=nodes,
                artifacts=artifacts,
                G=G,
                synth_prompt=synth_prompt,
            )

        return nodes, artifacts, G

    def _inject_for_artifact(
        self,
        artifact_name: str,
        artifact: Artifact,
        nodes: dict[str, Node],
        artifacts: dict[str, Artifact],
        G: nx.DiGraph,
        synth_prompt: str,
    ) -> tuple[dict[str, Node], dict[str, Artifact], nx.DiGraph]:

        synth_node_id = f"synthesizer_{artifact_name}"

        # ── Collect contributor system prompts ────────────────────────────────
        contributor_prompts = [
            nodes[c].system_prompt
            for c in artifact.contributors
            if c in nodes
        ]

        # ── Create partial artifacts — one per contributor ────────────────────
        partial_artifact_names: list[str] = []

        for contributor_id in artifact.contributors:
            partial_name = f"{artifact_name}__partial__{contributor_id}"
            partial_artifact = Artifact(
                name=partial_name,
                description=(
                    f"Partial contribution to '{artifact_name}' from agent '{contributor_id}'. "
                    f"Original artifact: {artifact.description}"
                ),
                contributors=[contributor_id],
                users=[synth_node_id],
                body={},
            )
            artifacts[partial_name] = partial_artifact
            partial_artifact_names.append(partial_name)

            # Update contributor node's output_artifacts to point to partial
            contributor_node = nodes[contributor_id]
            updated_outputs = [
                partial_name if a == artifact_name else a
                for a in contributor_node.output_artifacts
            ]
            nodes[contributor_id] = contributor_node.model_copy(
                update={"output_artifacts": updated_outputs}
            )

        # ── Create synthesizer node ───────────────────────────────────────────
        synth_node = Node(
            id=synth_node_id,
            name=f"Synthesizer: {artifact_name}",
            node_type=NodeType.SYNTHESIZER,
            system_prompt=synth_prompt,
            query_tool={},
            input_artifacts=partial_artifact_names,
            output_artifacts=[artifact_name],
            status=NodeStatus.PENDING,
            contributor_system_prompts=contributor_prompts,
        )
        nodes[synth_node_id] = synth_node

        # ── Update original artifact — now only synthesizer contributes ───────
        artifacts[artifact_name] = artifact.model_copy(
            update={"contributors": [synth_node_id]}
        )

        # ── Update DAG graph ──────────────────────────────────────────────────
        G.add_node(synth_node_id)

        # Edges from contributors → synthesizer
        for contributor_id in artifact.contributors:
            if contributor_id != synth_node_id:
                G.add_edge(contributor_id, synth_node_id, artifact=artifact_name)

        # Edges from synthesizer → original downstream users
        for user_id in artifact.users:
            G.add_edge(synth_node_id, user_id, artifact=artifact_name)

        # Remove old direct contributor → user edges for this artifact
        for contributor_id in artifact.contributors:
            for user_id in artifact.users:
                if G.has_edge(contributor_id, user_id):
                    G.remove_edge(contributor_id, user_id)

        return nodes, artifacts, G