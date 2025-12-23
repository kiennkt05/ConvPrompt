"""Storage utilities for RainbowPrompt tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        # Cache structure: layer_idx -> task_id -> {prompt, gate, key}
        self._cache: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

    @staticmethod
    def _compute_prompt_key(prompt: torch.Tensor) -> torch.Tensor:
        """Compute prompt key as mean of prompt tokens along prompt_length dimension.
        
        Args:
            prompt: Tensor of shape [prompt_length, embed_dim]
            
        Returns:
            Tensor of shape [embed_dim]
        """
        return prompt.mean(dim=0)

    def put(self, task_id: int, layer_idx: int, prompt: torch.Tensor, gate: torch.Tensor) -> None:
        """Store prompt/gate for a given task and layer, with computed prompt key.

        Args:
            task_id: Task identifier
            layer_idx: Layer index
            prompt: Prompt tensor of shape [prompt_length, embed_dim]
            gate: Gate tensor (scalar or small tensor)
        """
        if layer_idx not in self._cache:
            self._cache[layer_idx] = {}
        
        prompt_key = self._compute_prompt_key(prompt)
        
        self._cache[layer_idx][task_id] = {
            "prompt": prompt.detach().cpu().clone(),
            "gate": gate.detach().cpu().clone(),
            "key": prompt_key.detach().cpu().clone(),
        }

    def get(self, task_id: int, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve prompt/gate for a specific task and layer.

        Args:
            task_id: Task identifier
            layer_idx: Layer index
            
        Returns:
            Dict with keys "prompt", "gate", "key", or None if not found
        """
        layer_cache = self._cache.get(layer_idx)
        if layer_cache is None:
            return None
        
        stored = layer_cache.get(task_id)
        if stored is None:
            return None
        
        return {
            "prompt": stored["prompt"].clone(),
            "gate": stored["gate"].clone(),
            "key": stored["key"].clone(),
        }

    def get_all_prompts(self, layer_idx: int) -> List[Tuple[int, Dict[str, torch.Tensor]]]:
        """Retrieve all stored prompts for a layer, with their task_ids.

        Args:
            layer_idx: Layer index
            
        Returns:
            List of (task_id, {prompt, gate, key}) tuples for all stored tasks
        """
        layer_cache = self._cache.get(layer_idx)
        if layer_cache is None:
            return []
        
        result = []
        for task_id, stored in layer_cache.items():
            result.append((
                task_id,
                {
                    "prompt": stored["prompt"].clone(),
                    "gate": stored["gate"].clone(),
                    "key": stored["key"].clone(),
                }
            ))
        
        return result

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