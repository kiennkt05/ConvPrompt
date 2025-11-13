"""Rainbow prompt evolution module."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn


class RainbowEvolution(nn.Module):
    """Evolve base prompts into RainbowPrompts for a given task."""

    def __init__(
        self,
        embed_dim: int,
        prompt_length: int,
        proj_dim: int,
        align_hidden_dim: int,
        num_heads: int,
        use_task_conditioning: bool = True,
        enable_task_level: bool = True,
        enable_feature_level: bool = True,
        enable_alignment: bool = True,
        mode: str = "enhanced",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.prompt_length = prompt_length
        self.proj_dim = proj_dim
        self.align_hidden_dim = align_hidden_dim
        self.num_heads = num_heads
        self.use_task_conditioning = use_task_conditioning
        self.enable_task_level = enable_task_level
        self.enable_feature_level = enable_feature_level
        self.enable_alignment = enable_alignment
        self.mode = mode.lower()
        if self.mode not in {"enhanced", "paper"}:
            raise ValueError(f"Unsupported rainbow evolution mode: {mode}")

        self.task_proj = nn.Linear(embed_dim, proj_dim) if use_task_conditioning else None
        self.query_proj = nn.Linear(embed_dim, proj_dim)
        self.key_proj = nn.Linear(embed_dim, proj_dim)
        self.value_proj = nn.Linear(embed_dim, proj_dim)
        self.output_proj = nn.Linear(proj_dim, embed_dim)

        self.alignment = nn.Sequential(
            nn.Linear(embed_dim, align_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(align_hidden_dim, embed_dim),
        )

        self.layer_norm_in = nn.LayerNorm(embed_dim)
        self.layer_norm_out = nn.LayerNorm(embed_dim)

    def forward(
        self,
        base_prompts: torch.Tensor,
        new_prompt: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return evolved prompts and auxiliary statistics.

        Args:
            base_prompts: Tensor of shape [num_tasks, prompt_length, embed_dim].
                Includes prompts of all observed tasks (current task may be the
                last entry).
            new_prompt: Tensor of shape [prompt_length, embed_dim] for the
                current task. Used as the query when computing task-level
                relationships.
            task_embedding: Optional tensor of shape [embed_dim] representing
                the task descriptor embedding.

        Returns:
            Dict with keys:
                - "rainbow_prompt": evolved prompt to insert (shape
                  [prompt_length, embed_dim]).
                - "aligned_prompts": aligned prompts after evolution (shape
                  [num_tasks, prompt_length, embed_dim]).
                - "task_weights": attention weights over historical prompts
                  (shape [num_tasks]).
        """

        if base_prompts.ndim != 3:
            raise ValueError("base_prompts must have shape [num_tasks, prompt_length, embed_dim]")

        num_prompts, prompt_len, embed_dim = base_prompts.shape
        if prompt_len != self.prompt_length or embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected base_prompts with shape (*, {self.prompt_length}, {self.embed_dim}),"
                f" got {base_prompts.shape}."
            )

        if new_prompt.shape != (self.prompt_length, self.embed_dim):
            raise ValueError(
                f"new_prompt must have shape ({self.prompt_length}, {self.embed_dim}),"
                f" got {tuple(new_prompt.shape)}"
            )

        device = base_prompts.device
        task_weights = torch.ones(num_prompts, device=device) / max(num_prompts, 1)

        conditioned_prompts = base_prompts
        if self.use_task_conditioning and task_embedding is not None:
            task_vec = torch.tanh(self.task_proj(task_embedding))  # [proj_dim]
            key_repr = self.key_proj(base_prompts).mean(dim=1)  # [num_prompts, proj_dim]
            scores = torch.matmul(key_repr, task_vec) / math.sqrt(self.proj_dim)
            task_weights = torch.softmax(scores, dim=0)
            conditioned_prompts = base_prompts * (1 + task_weights[:, None, None])
        else:
            conditioned_prompts = base_prompts

        # Project prompts
        query = self.query_proj(new_prompt)  # [prompt_len, proj_dim]
        keys = self.key_proj(conditioned_prompts)  # [num_prompts, prompt_len, proj_dim]
        values = self.value_proj(conditioned_prompts)  # [num_prompts, prompt_len, proj_dim]

        # Task-level transformation: weight historical prompts for each layer
        if self.enable_task_level:
            pooled_query = query.mean(dim=0, keepdim=True)  # [1, proj_dim]
            pooled_keys = keys.mean(dim=1).transpose(0, 1)  # [proj_dim, num_prompts]
            logits = torch.matmul(pooled_query, pooled_keys) / math.sqrt(self.proj_dim)
            task_attn = torch.softmax(logits.squeeze(0), dim=-1)  # [num_prompts]

            if self.use_task_conditioning and task_embedding is not None:
                task_weights = 0.5 * task_weights + 0.5 * task_attn
            else:
                task_weights = task_attn

        task_weights = task_weights / task_weights.sum().clamp(min=1e-6)

        if self.mode == "enhanced":
            weighted_values = torch.einsum("p,pld->ld", task_weights, values)
            weighted_keys = torch.einsum("p,pld->ld", task_weights, keys)
            weighted_prompts = torch.einsum("p,pld->ld", task_weights, conditioned_prompts)

            # Feature-level transformation
            if self.enable_feature_level:
                feature_logits = torch.matmul(query, weighted_keys.transpose(0, 1)) / math.sqrt(self.proj_dim)
                feature_attn = torch.softmax(feature_logits, dim=-1)
                evolved_proj = torch.matmul(feature_attn, weighted_values)
            else:
                feature_attn = None
                evolved_proj = weighted_values

            # Project back to embedding dimension and add residual
            evolved_embeds = self.output_proj(evolved_proj)
            evolved_embeds = self.layer_norm_in(weighted_prompts + evolved_embeds)

            if self.enable_alignment:
                aligned = self.layer_norm_out(evolved_embeds + self.alignment(evolved_embeds))
            else:
                aligned = self.layer_norm_out(evolved_embeds)

            aligned_prompts = torch.stack([aligned for _ in range(num_prompts)], dim=0)

            feature_attn_out = feature_attn.detach() if feature_attn is not None else None
            rainbow_prompt = aligned
        else:
            # Paper-faithful evolution: evolve each prompt before averaging.
            if self.enable_feature_level:
                feature_logits = torch.matmul(
                    query.unsqueeze(0), keys.transpose(-2, -1)
                ) / math.sqrt(self.proj_dim)
                # feature_logits: [num_prompts, prompt_len, prompt_len]
                feature_attn = torch.softmax(feature_logits, dim=-1)
                evolved_proj = torch.matmul(feature_attn, values)
            else:
                feature_attn = None
                evolved_proj = values

            if self.enable_task_level:
                evolved_proj = evolved_proj * task_weights.view(num_prompts, 1, 1)

            evolved_embeds = self.output_proj(evolved_proj)
            evolved_embeds = self.layer_norm_in(conditioned_prompts + evolved_embeds)

            if self.enable_alignment:
                aligned_prompts = self.layer_norm_out(
                    evolved_embeds + self.alignment(evolved_embeds)
                )
            else:
                aligned_prompts = self.layer_norm_out(evolved_embeds)

            rainbow_prompt = aligned_prompts.mean(dim=0)
            feature_attn_out = (
                feature_attn.detach().mean(dim=0)
                if feature_attn is not None
                else None
            )

        return {
            "rainbow_prompt": rainbow_prompt,
            "aligned_prompts": aligned_prompts,
            "task_weights": task_weights.detach(),
            "feature_attn": feature_attn_out,
        }

