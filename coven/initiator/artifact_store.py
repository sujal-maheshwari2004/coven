from __future__ import annotations

import asyncio
from typing import Any

from coven.models import Artifact


class ArtifactStore:
    """
    Shared in-memory store for all artifacts produced during DAG execution.

    Acts as the single source of truth for artifact state across all levels.
    Thread-safe via asyncio.Lock — multiple agents in the same level
    can safely write their outputs concurrently.

    Lifecycle:
    - Initialized with all artifact definitions at DAG start (bodies empty)
    - Agents write their output artifact bodies via put()
    - Downstream agents read their input artifacts via get()
    - At DAG completion the compiler reads all final artifacts
    """

    def __init__(self, artifacts: dict[str, Artifact]):
        self._store: dict[str, Artifact] = dict(artifacts)
        self._lock = asyncio.Lock()

    async def get(self, artifact_name: str) -> Artifact:
        """
        Retrieve an artifact by name.

        Args:
            artifact_name: The artifact's unique name.

        Returns:
            The Artifact model (may have empty body if not yet produced).

        Raises:
            KeyError: If artifact name is not registered in the store.
        """
        async with self._lock:
            if artifact_name not in self._store:
                raise KeyError(
                    f"Artifact '{artifact_name}' not found in store. "
                    f"Registered artifacts: {list(self._store.keys())}"
                )
            return self._store[artifact_name]

    async def put(self, artifact_name: str, body: dict[str, Any]) -> None:
        """
        Write a produced artifact body into the store.

        Args:
            artifact_name: The artifact's unique name.
            body: The produced artifact body from an agent.

        Raises:
            KeyError: If artifact name is not registered in the store.
        """
        async with self._lock:
            if artifact_name not in self._store:
                raise KeyError(
                    f"Cannot write artifact '{artifact_name}' — not registered in store."
                )
            existing = self._store[artifact_name]
            self._store[artifact_name] = existing.model_copy(
                update={"body": body}
            )

    async def get_many(self, artifact_names: list[str]) -> list[Artifact]:
        """
        Retrieve multiple artifacts by name in one call.

        Args:
            artifact_names: List of artifact names to retrieve.

        Returns:
            List of Artifact models in the same order as input names.
        """
        return [await self.get(name) for name in artifact_names]

    async def all(self) -> dict[str, Artifact]:
        """
        Return a snapshot of the entire artifact store.
        Used by the compiler at the end of execution.
        """
        async with self._lock:
            return dict(self._store)

    def is_ready(self, artifact_name: str) -> bool:
        """
        Check if an artifact has a non-empty body (i.e. has been produced).

        Args:
            artifact_name: The artifact's unique name.

        Returns:
            True if the artifact body is non-empty.
        """
        artifact = self._store.get(artifact_name)
        if artifact is None:
            return False
        return bool(artifact.body)

    def all_inputs_ready(self, input_artifact_names: list[str]) -> bool:
        """
        Check if all input artifacts for a node are ready to consume.

        Args:
            input_artifact_names: List of artifact names the node needs.

        Returns:
            True if all are ready.
        """
        return all(self.is_ready(name) for name in input_artifact_names)