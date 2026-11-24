"""Map one traffic sensor to one logical client without changing FATE parties.

This module only describes the layout. Trainers must explicitly opt in to
batched logical-client transport before using it for training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


PEMS_FILES = {
    "PeMS03": "data/PeMS03/pems03.npz",
    "PeMS04": "data/PeMS04/pems04.npz",
    "PeMSD7": "data/PeMSD7/pemsd7.npz",
    "PeMS08": "data/PeMS08/pems08.npz",
}


def dataset_node_count(project_root: str | Path, dataset_name: str) -> int:
    """Read the node axis from the actual dataset, not a hard-coded count."""
    try:
        relative_path = PEMS_FILES[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Node-client layout does not support {dataset_name}") from exc
    with np.load(Path(project_root) / relative_path, mmap_mode="r") as archive:
        shape = archive["data"].shape
    if len(shape) < 2 or shape[1] < 1:
        raise ValueError(f"Invalid node axis in {relative_path}: {shape}")
    return int(shape[1])


def assign_logical_clients(node_count: int, worker_count: int = 4) -> list[list[int]]:
    """Assign every node to exactly one physical client worker."""
    if not isinstance(node_count, int) or not isinstance(worker_count, int):
        raise TypeError("node_count and worker_count must be integers")
    if node_count < 1 or worker_count < 1 or worker_count > node_count:
        raise ValueError("Require 1 <= worker_count <= node_count")
    quotient, remainder = divmod(node_count, worker_count)
    assignments = []
    start = 0
    for worker in range(worker_count):
        stop = start + quotient + (worker < remainder)
        assignments.append(list(range(start, stop)))
        start = stop
    return assignments
