from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ArtifactStatus = Literal[
    "pending", "running", "completed", "failed", "gate_failed", "skipped"
]


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    experiment: str
    dataset: str
    split_hash: str
    model: str
    conditioner: str
    seed: int
    checkpoint: str
    result_json: str
    figure_paths: list[str]
    status: ArtifactStatus
    notes: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this record."""
        return asdict(self)
