"""Storage utilities for RainbowPrompt tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch


class RainbowPromptStorage:
    """In-memory store for RainbowPrompts per task and layer.

    NOTE:
        This implementation is intentionally kept in-memory only. It does
        not perform any on-disk serialization, and is meant to keep the
        latest prompts alive only for the current run.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # Single global cache: we keep only the latest prompts per layer for
        # the current run, independent of task id.
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def put(self, task_id: int, layer_idx: int, prompt: torch.Tensor, gate: torch.Tensor) -> None:
        """Store the latest prompt/gate for a given layer, ignoring task_id.

        The `task_id` argument is retained only for API compatibility; this
        storage keeps a single global RainbowPrompt per layer.
        """
        _ = task_id  # unused
        self._cache[layer_idx] = {
            "prompt": prompt.detach().cpu().clone(),
            "gate": gate.detach().cpu().clone(),
        }

    def get(self, task_id: int, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve the latest global prompt/gate for a layer, ignoring task_id.

        The `task_id` argument is retained only for API compatibility; prompts
        are stored per-layer globally, corresponding to the most recently
        finalized RainbowPrompt in this run.
        """
        _ = task_id  # unused
        stored = self._cache.get(layer_idx)
        if stored is None:
            return None
        return {
            "prompt": stored["prompt"].clone(),
            "gate": stored["gate"].clone(),
        }

    def save_task(self, task_id: int) -> None:
        """No-op for compatibility.

        Historically this method serialized prompts for a task to disk.
        We now keep prompts in-memory only and do not distinguish tasks, so
        this is intentionally a no-op to preserve the public API without
        performing I/O.
        """
        _ = task_id  # unused

    def load_task(self, task_id: int, device: torch.device | None = None) -> None:
        """No-op for compatibility.

        Disk-based loading of prompts is no longer supported, and prompts are
        not partitioned by task. The method remains to avoid breaking existing
        call sites but does not modify the in-memory cache.
        """
        _ = task_id  # unused
        _ = device  # unused, kept for signature compatibility
        return