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
        use_paper_evolution: bool = False,
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
        self.use_paper_evolution = use_paper_evolution

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

        # Task conditioning: create attention-weighted combination of base prompts
        if self.use_task_conditioning and task_embedding is not None:
            # Paper: Task conditioning via attention (Section 3.2.1)
            # σ(e_t) P_l / sqrt(d_p) -> softmax -> attention weights
            # Then: P_l <- softmax(...) P_l (weighted combination)
            task_vec = torch.sigmoid(self.task_proj(task_embedding))  # [proj_dim], using sigmoid as in paper
            # Project base_prompts to compute similarity
            projected_prompts = self.key_proj(base_prompts)  # [num_prompts, prompt_len, proj_dim]
            # Compute similarity per prompt (average over prompt_len for task-level similarity)
            key_repr = projected_prompts.mean(dim=1)  # [num_prompts, proj_dim]
            scores = torch.matmul(key_repr, task_vec) / math.sqrt(self.proj_dim)  # [num_prompts]
            task_weights_conditioning = torch.softmax(scores, dim=0)  # [num_prompts]
            # Paper: Attention-based combination - weighted sum of prompts
            # This creates a single conditioned prompt from all base prompts
            conditioned_single = torch.einsum("p,pld->ld", task_weights_conditioning, base_prompts)  # [prompt_len, embed_dim]
            # Expand to maintain shape [num_prompts, prompt_len, embed_dim] for downstream processing
            # All prompts become the same conditioned prompt (as per paper's attention-based combination)
            conditioned_prompts = conditioned_single.unsqueeze(0).expand(num_prompts, -1, -1)  # [num_prompts, prompt_len, embed_dim]
        else:
            conditioned_prompts = base_prompts

        # Project prompts
        query = self.query_proj(new_prompt)  # [prompt_len, proj_dim]
        keys = self.key_proj(conditioned_prompts)  # [num_prompts, prompt_len, proj_dim]
        values = self.value_proj(conditioned_prompts)  # [num_prompts, prompt_len, proj_dim]

        feature_attn: Optional[torch.Tensor]

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

        if not self.use_paper_evolution:
            weighted_values = torch.einsum("p,pld->ld", task_weights, values)
            weighted_keys = torch.einsum("p,pld->ld", task_weights, keys)
            weighted_prompts = torch.einsum("p,pld->ld", task_weights, conditioned_prompts)

            if self.enable_feature_level:
                # Paper: Feature-level attention (Q^T K / sqrt(d_k))
                # Q: [prompt_len, proj_dim] -> transpose -> [proj_dim, prompt_len]
                # K: [prompt_len, proj_dim] -> transpose -> [proj_dim, prompt_len]
                # Q^T @ K: [proj_dim, prompt_len] @ [prompt_len, proj_dim] = [proj_dim, proj_dim]
                feature_logits = torch.matmul(query.transpose(-1, -2), weighted_keys) / math.sqrt(self.proj_dim)  # [proj_dim, proj_dim]
                feature_attn = torch.softmax(feature_logits, dim=-1)  # [proj_dim, proj_dim]
                # Apply attention to values: feature_attn @ V^T
                # V: [prompt_len, proj_dim] -> transpose -> [proj_dim, prompt_len]
                # feature_attn @ V^T: [proj_dim, proj_dim] @ [proj_dim, prompt_len] = [proj_dim, prompt_len]
                # Transpose back: [prompt_len, proj_dim]
                evolved_proj = torch.matmul(feature_attn, weighted_values.transpose(-1, -2)).transpose(-1, -2)  # [prompt_len, proj_dim]
            else:
                feature_attn = None
                evolved_proj = weighted_values

            evolved_embeds = self.output_proj(evolved_proj)
            evolved_embeds = self.layer_norm_in(weighted_prompts + evolved_embeds)

            if self.enable_alignment:
                aligned = self.layer_norm_out(evolved_embeds + self.alignment(evolved_embeds))
            else:
                aligned = self.layer_norm_out(evolved_embeds)

            aligned_prompts = torch.stack([aligned for _ in range(num_prompts)], dim=0)

            return {
                "rainbow_prompt": aligned,
                "aligned_prompts": aligned_prompts,
                "task_weights": task_weights.detach(),
                "feature_attn": feature_attn.detach() if feature_attn is not None else None,
            }

        expanded_weights = task_weights.view(num_prompts, 1, 1)
        weighted_values = expanded_weights * values
        weighted_keys = expanded_weights * keys

        aligned_prompts_list = []
        feature_attn_list = []

        for prompt_idx in range(num_prompts):
            value_i = weighted_values[prompt_idx]
            key_i = weighted_keys[prompt_idx]
            prompt_i = conditioned_prompts[prompt_idx]

            if self.enable_feature_level:
                # Paper: Feature-level attention (Q^T K / sqrt(d_k))
                # Q: [prompt_len, proj_dim] -> transpose -> [proj_dim, prompt_len]
                # K: [prompt_len, proj_dim] -> transpose -> [proj_dim, prompt_len]
                # Q^T @ K: [proj_dim, prompt_len] @ [prompt_len, proj_dim] = [proj_dim, proj_dim]
                logits = torch.matmul(query.transpose(-1, -2), key_i) / math.sqrt(self.proj_dim)  # [proj_dim, proj_dim]
                attn = torch.softmax(logits, dim=-1)  # [proj_dim, proj_dim]
                # Apply attention to values: attn @ V^T
                # V: [prompt_len, proj_dim] -> transpose -> [proj_dim, prompt_len]
                # attn @ V^T: [proj_dim, proj_dim] @ [proj_dim, prompt_len] = [proj_dim, prompt_len]
                # Transpose back: [prompt_len, proj_dim]
                evolved_proj = torch.matmul(attn, value_i.transpose(-1, -2)).transpose(-1, -2)  # [prompt_len, proj_dim]
                feature_attn_list.append(attn.detach())
            else:
                evolved_proj = value_i

            evolved_embeds = self.output_proj(evolved_proj)
            evolved_embeds = self.layer_norm_in(prompt_i + evolved_embeds)

            if self.enable_alignment:
                aligned_i = self.layer_norm_out(evolved_embeds + self.alignment(evolved_embeds))
            else:
                aligned_i = self.layer_norm_out(evolved_embeds)

            aligned_prompts_list.append(aligned_i)

        aligned_prompts = torch.stack(aligned_prompts_list, dim=0)
        rainbow_prompt = aligned_prompts.mean(dim=0)
        feature_attn = torch.stack(feature_attn_list, dim=0) if feature_attn_list else None

        return {
            "rainbow_prompt": rainbow_prompt,
            "aligned_prompts": aligned_prompts,
            "task_weights": task_weights.detach(),
            "feature_attn": feature_attn,
        }

