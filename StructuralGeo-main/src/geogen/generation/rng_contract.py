"""Order-independent named random streams for reproducible geology generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np


RNG_CONTRACT_VERSION = "structuralgeo_named_seedsequence_v1"


def _component_words(component: str) -> tuple[int, int, int, int]:
    """Map a namespace component to stable uint32 SeedSequence spawn words."""
    if not component or component.strip() != component:
        raise ValueError(f"invalid RNG namespace component: {component!r}")
    digest = hashlib.sha256(component.encode("utf-8")).digest()[:16]
    return tuple(int.from_bytes(digest[index : index + 4], "little") for index in range(0, 16, 4))


@dataclass(frozen=True)
class NamedSeedSequence:
    """A root seed plus a stable namespace for deriving independent RNG streams.

    Child derivation is based on ``SeedSequence.spawn_key`` words obtained from
    SHA-256 namespace components. It is independent of call order and Python's
    randomized hash implementation.
    """

    root_seed: int
    namespace: tuple[str, ...] = ()
    version: str = RNG_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if int(self.root_seed) < 0:
            raise ValueError("root_seed must be non-negative")
        if self.version != RNG_CONTRACT_VERSION:
            raise ValueError(f"unsupported RNG contract version: {self.version}")
        for component in self.namespace:
            _component_words(component)

    @classmethod
    def root(cls, root_seed: int) -> "NamedSeedSequence":
        """Create the single root SeedSequence contract for a generated case."""
        return cls(root_seed=int(root_seed))

    def child(self, *components: str) -> "NamedSeedSequence":
        if not components:
            raise ValueError("at least one child namespace component is required")
        return NamedSeedSequence(
            root_seed=self.root_seed,
            namespace=self.namespace + tuple(str(component) for component in components),
            version=self.version,
        )

    def seed_sequence(self, stream_name: str) -> np.random.SeedSequence:
        components: Sequence[str] = (self.version, *self.namespace, str(stream_name))
        spawn_key = tuple(word for component in components for word in _component_words(component))
        return np.random.SeedSequence(entropy=int(self.root_seed), spawn_key=spawn_key)

    def generator(self, stream_name: str) -> np.random.Generator:
        return np.random.default_rng(self.seed_sequence(stream_name))

    def uint32_seed(self, stream_name: str) -> int:
        state = self.seed_sequence(stream_name).generate_state(1, dtype=np.uint32)
        return int(state[0])

    def describe(self) -> dict[str, object]:
        return {
            "version": self.version,
            "root_seed": int(self.root_seed),
            "namespace": list(self.namespace),
            "derivation": "SeedSequence(root_seed, spawn_key=SHA256(component)[0:16] as 4 little-endian uint32 words)",
            "bit_generator": "PCG64",
        }
