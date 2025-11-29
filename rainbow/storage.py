"""Storage utilities for RainbowPrompt tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch


class RainbowPromptStorage:
    """Store and retrieve RainbowPrompts per task and layer."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

    def put(self, task_id: int, layer_idx: int, prompt: torch.Tensor, gate: torch.Tensor) -> None:
        if task_id not in self._cache:
            self._cache[task_id] = {}
        self._cache[task_id][layer_idx] = {
            "prompt": prompt.detach().cpu().clone(),
            "gate": gate.detach().cpu().clone(),
        }

    def get(self, task_id: int, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if task_id not in self._cache:
            file_path = self.root / f"task_{task_id:03d}.pt"
            if file_path.exists():
                data = torch.load(file_path, map_location="cpu")
                self._cache[task_id] = {int(k): {"prompt": v["prompt"], "gate": v["gate"]} for k, v in data.items()}
            else:
                return None

        stored = self._cache[task_id].get(layer_idx)
        if stored is None:
            return None
        return {
            "prompt": stored["prompt"].clone(),
            "gate": stored["gate"].clone(),
        }

    def save_task(self, task_id: int) -> None:
        if task_id not in self._cache:
            raise KeyError(f"No prompts cached for task {task_id}")

        serialized = {
            layer: {"prompt": data["prompt"].cpu(), "gate": data["gate"].cpu()}
            for layer, data in self._cache[task_id].items()
        }
        file_path = self.root / f"task_{task_id:03d}.pt"
        torch.save(serialized, file_path)

    def load_task(self, task_id: int, device: torch.device | None = None) -> None:
        file_path = self.root / f"task_{task_id:03d}.pt"
        if not file_path.exists():
            raise FileNotFoundError(f"RainbowPrompts for task {task_id} not found at {file_path}")

        data = torch.load(file_path, map_location=device or "cpu")
        self._cache[task_id] = {
            int(k): {
                "prompt": v["prompt"].to(device or "cpu"),
                "gate": v["gate"].to(device or "cpu"),
            }
            for k, v in data.items()
        }

    # NOTE: The RainbowPromptModule uses the internal `_cache` to build
    # multi-task prompts at inference time. We intentionally keep this
    # class lightweight and do not add extra public aggregation helpers
    # here; see `RainbowPromptModule._prepare_inference_prompt` for how
    # the cached tensors are combined across tasks.

