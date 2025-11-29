import copy
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rainbow.evolution import RainbowEvolution
from rainbow.gating import ProbabilisticGate
from rainbow.storage import RainbowPromptStorage


class RainbowPromptModule(nn.Module):
    """Manage RainbowPrompt evolution, gating, and storage across ViT layers."""

    def __init__(
        self,
        embed_dim: int,
        prompt_length: int,
        num_layers: int,
        num_heads: int,
        proj_dim: int,
        align_hidden_dim: int,
        gate_tau_start: float,
        gate_tau_end: float,
        gate_harden_at: float,
        save_dir: str,
        use_task_conditioning: bool = True,
        enable_task_level: bool = True,
        enable_feature_level: bool = True,
        enable_alignment: bool = True,
        use_adaptive_gating: bool = True,
        use_paper_evolution: bool = False,
        hard_prompt_selection: bool = False,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads for prefix prompts")

        self.embed_dim = embed_dim
        self.prompt_length = prompt_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_task_conditioning = use_task_conditioning
        self.use_adaptive_gating = use_adaptive_gating
        self.use_paper_evolution = use_paper_evolution
        # If True, use a single task's prompt at inference (hard routing) instead
        # of concatenating prompts from all tasks (soft routing).
        self.hard_prompt_selection = hard_prompt_selection

        self.evolutions = nn.ModuleList(
            [
                RainbowEvolution(
                    embed_dim=embed_dim,
                    prompt_length=prompt_length,
                    proj_dim=proj_dim,
                    align_hidden_dim=align_hidden_dim,
                    num_heads=num_heads,
                    use_task_conditioning=use_task_conditioning,
                    enable_task_level=enable_task_level,
                    enable_feature_level=enable_feature_level,
                    enable_alignment=enable_alignment,
                    use_paper_evolution=use_paper_evolution,
                )
                for _ in range(num_layers)
            ]
        )

        self.base_prompts = nn.ModuleList([nn.ParameterList() for _ in range(num_layers)])
        self.storage = RainbowPromptStorage(save_dir)

        self.current_task_id: Optional[int] = None
        self.current_gate: Optional[ProbabilisticGate] = None
        self.current_epoch: int = 0
        self.max_epochs: int = 1
        self.task_embedding: Optional[torch.Tensor] = None
        self.training_mode: bool = True

        self._latest_layer_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._aux_losses: Dict[str, torch.Tensor] = {}

        self._gate_config = dict(
            tau_start=gate_tau_start,
            tau_end=gate_tau_end,
            harden_epoch_ratio=gate_harden_at,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_epoch(self, epoch: int, max_epochs: int) -> None:
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def update_task_embedding(self, embedding: Optional[torch.Tensor]) -> None:
        self.task_embedding = embedding

    def set_training(self, mode: bool) -> None:
        self.training_mode = mode

    def start_task(self, task_id: int) -> None:
        """Initialize base prompts and gating for a new task."""
        self.current_task_id = task_id
        device = self.device

        for layer_idx in range(self.num_layers):
            for prompt in self.base_prompts[layer_idx]:
                prompt.requires_grad = False

            new_prompt = nn.Parameter(torch.zeros(self.prompt_length, self.embed_dim, device=device))
            nn.init.normal_(new_prompt, mean=0.0, std=0.02)
            self.base_prompts[layer_idx].append(new_prompt)

        if self.use_adaptive_gating:
            self.current_gate = ProbabilisticGate(num_layers=self.num_layers, **self._gate_config).to(device)
        else:
            self.current_gate = None
        self._latest_layer_cache.clear()
        self._aux_losses.clear()

    def _stack_prompts(self, layer_idx: int) -> torch.Tensor:
        prompts = list(self.base_prompts[layer_idx])
        if not prompts:
            raise ValueError(f"No base prompts registered for layer {layer_idx}")
        return torch.stack([p for p in prompts], dim=0)

    def _format_prompt(self, prompt: torch.Tensor, gate_value: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Format a (possibly multi-task) prompt for PreT_Attention.

        The first dimension of `prompt` is treated as the *effective* prompt length,
        which allows us to concatenate prompts from multiple tasks at inference time
        (soft routing), while still supporting single-task prompts during training.
        """
        prompt_length = prompt.shape[0]
        prompt = prompt.view(prompt_length, self.num_heads, self.head_dim)
        key_prompt = prompt
        value_prompt = prompt
        prefix = torch.stack([key_prompt, value_prompt], dim=0)  # [2, length, num_heads, head_dim]
        prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).contiguous()
        gate_scale = gate_value.view(1, 1, 1, 1, 1)
        return prefix * gate_scale

    def _prepare_training_prompt(self, layer_idx: int, batch_size: int) -> Optional[torch.Tensor]:
        if self.current_gate is None:
            raise RuntimeError("Probabilistic gate not initialized. Call start_task() first.")

        base_prompts = self._stack_prompts(layer_idx)
        new_prompt = self.base_prompts[layer_idx][-1]

        task_embedding = None
        if self.use_task_conditioning and self.task_embedding is not None:
            task_embedding = self.task_embedding.to(base_prompts.device)

        evo_out = self.evolutions[layer_idx](base_prompts, new_prompt, task_embedding)
        rainbow_prompt = evo_out["rainbow_prompt"]

        if self.use_adaptive_gating:
            if self.current_gate is None:
                raise RuntimeError("Probabilistic gate not initialized. Call start_task() first.")

            gate_out = self.current_gate(
                layer_idx=layer_idx,
                epoch=self.current_epoch,
                max_epochs=self.max_epochs,
                training=self.training_mode,
            )

            gate_value = gate_out["gate"].to(rainbow_prompt.device)
            self._aux_losses[f"sparsity_{layer_idx}"] = gate_out["sparsity_loss"]
        else:
            gate_value = torch.ones((), device=rainbow_prompt.device)

        formatted_prompt = self._format_prompt(rainbow_prompt, gate_value, batch_size)

        self._latest_layer_cache[layer_idx] = {
            "prompt": rainbow_prompt.detach().cpu(),
            "gate": gate_value.detach().cpu(),
        }

        return formatted_prompt

    def _prepare_inference_prompt(self, layer_idx: int, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """Build a multi-task prefix prompt for inference.

        Instead of selecting a single task's prompt (via task_id), we concatenate
        prompts from *all* observed tasks along the length dimension. The ViT
        attention then performs soft routing over these positions automatically.
        """
        # Ensure any on-disk prompts are loaded into the storage cache.
        # This is a no-op if everything is already cached.
        if not self.storage._cache:  # type: ignore[attr-defined]
            # Lazy import to avoid circular imports at module load time.
            from pathlib import Path

            root: Path = self.storage.root  # type: ignore[attr-defined]
            for path in root.glob("task_*.pt"):
                try:
                    task_str = path.stem.split("_")[1]
                    task_idx = int(task_str)
                except (IndexError, ValueError):
                    continue
                # Load without overriding device placement logic inside storage.
                self.storage.load_task(task_idx, device=device)

        prompts = []
        # Sort by task id to keep ordering consistent across runs.
        for task_id in sorted(self.storage._cache.keys()):  # type: ignore[attr-defined]
            layer_dict = self.storage._cache[task_id]  # type: ignore[index]
            if layer_idx not in layer_dict:
                continue
            data = layer_dict[layer_idx]
            prompts.append(data["prompt"].to(device))

        if not prompts:
            return None

        # Concatenate prompts from all tasks along the length dimension:
        # [T, L, D] -> [T*L, D]
        prompt_tensor = torch.cat(prompts, dim=0)
        gate_value = torch.ones((), device=device)
        return self._format_prompt(prompt_tensor, gate_value, batch_size)

    def _prepare_single_task_inference_prompt(
        self,
        task_id: int,
        layer_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Inference prefix that uses only one task's prompt (hard routing).

        This mirrors the original behavior where a single task_id selects the
        corresponding stored prompt for each layer, similar to DualPrompt's
        hard E-Prompt selection.
        """
        stored = self.storage.get(task_id, layer_idx)
        if stored is None:
            return None

        prompt_tensor = stored["prompt"].to(device)
        gate_value = stored["gate"].to(device)
        return self._format_prompt(prompt_tensor, gate_value, batch_size)

    def forward(
        self,
        task_id: int,
        layer_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.training_mode:
            # During training we only expose the current task's prompt;
            # routing is trivial because there is exactly one prompt per layer.
            return self._prepare_training_prompt(layer_idx, batch_size)

        # Inference-time routing:
        # - hard_prompt_selection=True  -> use only the prompt for `task_id`
        # - hard_prompt_selection=False -> expose prompts from all observed tasks
        #   so that self-attention can route softly across them.
        if self.hard_prompt_selection:
            return self._prepare_single_task_inference_prompt(task_id, layer_idx, batch_size, device)
        return self._prepare_inference_prompt(layer_idx, batch_size, device)

    def auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        losses = {k: v for k, v in self._aux_losses.items()}
        self._aux_losses.clear()
        return losses

    def finalize_task(self, task_id: int) -> None:
        if not self._latest_layer_cache:
            return
        for layer_idx, data in self._latest_layer_cache.items():
            self.storage.put(task_id, layer_idx, data["prompt"], data["gate"])
        self.storage.save_task(task_id)
        self._latest_layer_cache.clear()

    def load_task(self, task_id: int, device: torch.device) -> None:
        self.storage.load_task(task_id, device=device)


def get_out_batch(batch_size, task_mean, task_std):
    out = []
    for i in range(batch_size):
        out.append(task_mean + task_std * torch.randn_like(task_mean))
    return torch.stack(out).to(task_mean.device)

class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, num_tasks=10, kernel_size=17, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,
                 prompts_per_task=5):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.top_k = top_k
        self.n_tasks = num_tasks
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.use_prompt_embed_matcher = True 
        self.prompts_per_task = prompts_per_task
        self.old_num_k = 0
        self.new_num_k = 0
        self.sigmoid = nn.Sigmoid()

        self.ker_size = kernel_size
        self.stride = 1
        self.dilation = 1
        self.conv_channels = 1 


        # This is the max number of kernels (self.pool_size) that can occur if we use max number of prompts per task
        self.pool_size = self.n_tasks * prompts_per_task
        print("Num Tasks: ", self.n_tasks, "pool_size: ", self.pool_size, "kernel_size: ", kernel_size, "top_k: ", top_k)
        if self.use_prompt_embed_matcher:
            self.prompt_embed_matcher = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(embed_dim, embed_dim // 2)),
                ('relu1', nn.ReLU()),
                ('linear2', nn.Linear(embed_dim // 2, embed_dim // 4))
            ]))
        self.k_conv_vals = nn.ModuleDict()
        for i in range(self.num_layers):
            self.k_conv_vals[str(i)] = nn.ModuleDict()
            for j in range(self.num_heads):
                self.k_conv_vals[str(i)][str(j)] = nn.ModuleList()
                for k in range(self.pool_size):
                    conv_val = nn.Conv2d(1, 1, (self.ker_size, self.ker_size),  \
                                         stride=self.stride, dilation=self.dilation)
                    # conv_val.weight.data = torch.nn.init.normal_(conv_val.weight.data)
                    self.k_conv_vals[str(i)][str(j)].append(conv_val)
    
        

        self.v_conv_vals = nn.ModuleDict()
        for i in range(self.num_layers):
            self.v_conv_vals[str(i)] = nn.ModuleDict()
            for j in range(self.num_heads):
                self.v_conv_vals[str(i)][str(j)] = nn.ModuleList()
                for k in range(self.pool_size):

                    conv_val = nn.Conv2d(1, 1, (self.ker_size, self.ker_size), \
                                         stride=self.stride, dilation=self.dilation)
                    # conv_val.weight.data = torch.nn.init.normal_(conv_val.weight.data)
                    self.v_conv_vals[str(i)][str(j)].append(conv_val)

            # user prefix style
        if self.use_prefix_tune_for_e_prompt:
            assert embed_dim % self.num_heads == 0
            if self.same_key_value:
                prompt_pool_shape = (self.num_layers, self.length + self.ker_size - 1, 
                                    embed_dim + (self.ker_size - 1) * self.num_heads)

                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
            else:
                prompt_pool_shape = (self.num_layers, (self.length + self.ker_size - 1) * 2, 
                                    embed_dim + (self.ker_size - 1) * self.num_heads)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                    nn.init.uniform_(self.prompt, -1, 1)
                elif prompt_init == 'ortho':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    for i in range(self.num_layers):
                        nn.init.orthogonal_(self.prompt[i])

        else:
            prompt_pool_shape=(self.num_layers, self.length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
            elif prompt_init == 'ortho':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.orthogonal_(self.prompt)
                    
        # if using learnable prompt keys
        if prompt_key:
            if self.use_prompt_embed_matcher:
                key_shape = (self.pool_size, embed_dim // 4)
            else:
                key_shape = (self.pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
            elif prompt_key_init == 'ortho':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.orthogonal_(self.prompt_key)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean 
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def process_new_task(self, old_num_k, new_num_k):
        self.old_num_k = old_num_k
        self.new_num_k = new_num_k
        print("Old Num K: ", self.old_num_k, "New Num K: ", self.new_num_k)
        self.prompt_key = self.gram_schmidt(self.prompt_key)


    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point

        s = self.old_num_k
        f = self.new_num_k
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_embed, task_id=-1, prompt_mask=None, layer_num= -1, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            # pt = int()
            # s = int(self.task_count * self.pool_size / (self.n_tasks))
            # f = int((self.task_count + 1) * self.pool_size / (self.n_tasks))
            s = self.old_num_k
            f = self.new_num_k

            if self.training:
                if task_id > 0:
                    prompt_key = torch.cat((self.prompt_key[:s].detach().clone(), self.prompt_key[s:f]), dim=0)                   
                else:
                    prompt_key = self.prompt_key[s:f]
            else:
                prompt_key = self.prompt_key[0:f]


            if self.use_prompt_embed_matcher:
                x_embed_mean = self.prompt_embed_matcher(x_embed_mean)

            # prompt_key = self.sigmoid(prompt_key)
            # x_embed = self.sigmoid(x_embed)
            prompt_key_norm = self.l2_normalize(prompt_key, dim=-1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1) # B, C
            

            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
            similarity = similarity.t() # B, pool_size
            # similarity
            out['similarity'] = similarity
            # print('weighting score: ', similarity)

            if self.use_prefix_tune_for_e_prompt:
                # print("Prompt Shape = ", self.prompt.shape)

                # exit()
                batched_prompt_raw = self.prompt[layer_num].unsqueeze(0).repeat(x_embed.shape[0], 1, 1)  #  B, length, C
                dual = 2
                batch_size, length, embed_dim = batched_prompt_raw.shape
                # print("Batched Prompt Shape = ", batched_prompt_raw.shape)
                batched_prompt = batched_prompt_raw.reshape(
                    batch_size, dual, length // dual, self.num_heads, embed_dim // self.num_heads
                )
                # print("Batched Prompt Shape = ", batched_prompt.shape)
                # exit()
            else:
                batched_prompt_raw = self.prompt[layer_num,:]
                batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    batch_size, top_k * length, embed_dim
                )

            




            # out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm



        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0

                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)

                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)

            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)
        



        batched_prompt = self.compute_conv_over_prompt(batched_prompt, f, layer_num, similarity)


        out['batched_prompt'] = batched_prompt

        return out


    def ortho_penalty(self, t):
        return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean() * 1e-6


    def compute_conv_over_prompt(self, batched_prompt, f, layer_num, similarity):
        #  batch_size, dual, length // dual, self.num_heads, embed_dim // self.num_heads
        batched_prompt = batched_prompt.permute(1, 3, 0, 2, 4) #  dual, num_heads, B, length, head_dim
        k_prompt_list = []
        v_prompt_list = []
        
        k_prompt_layer = batched_prompt[0] #  num_heads, B,  length, head_dim
        v_prompt_layer = batched_prompt[1] # num_heads, B,length, head_dim
        n_heads, batch_size, length, head_dim = k_prompt_layer.shape
        # print("K prompt layer shape: ", k_prompt_layer.shape)
        length = length - self.ker_size + 1
        head_dim = head_dim - self.ker_size + 1
        new_k_prompt_layer = torch.zeros((n_heads, batch_size, length, head_dim), device=k_prompt_layer.device)
        new_v_prompt_layer = torch.zeros((n_heads, batch_size, length, head_dim), device=k_prompt_layer.device)
        for h in range(self.num_heads):
            k_conv_vals = self.k_conv_vals[str(layer_num)][str(h)]
            v_conv_vals = self.v_conv_vals[str(layer_num)][str(h)]
            k_prompt_head = k_prompt_layer[h].unsqueeze(1) #  B, 1, length, head_dim
            v_prompt_head = v_prompt_layer[h].unsqueeze(1) # B, 1, length, head_dim
            for p in range(f):
                k_conv_val = k_conv_vals[p]
                v_conv_val = v_conv_vals[p]
                new_k_prompt_layer[h] += k_conv_val(k_prompt_head).squeeze(1) * similarity[:, p].unsqueeze(1).unsqueeze(2)
                new_v_prompt_layer[h] += v_conv_val(v_prompt_head).squeeze(1)  * similarity[:, p].unsqueeze(1).unsqueeze(2)
            # k_prompt_list.append(new_k_prompt_layer)  # num_layers, num_heads, B,length, head_dim
            # v_prompt_list.append(new_v_prompt_layer)
        # new_k_prompt = torch.stack(k_prompt_list, dim=0) # num_layers, num_heads, B,length, head_dim
        # new_v_prompt = torch.stack(v_prompt_list, dim=0) # num_layers, num_heads, B,length, head_dim
        new_batched_prompt = torch.stack([new_k_prompt_layer, new_v_prompt_layer], dim=0) # dual, num_heads, B, length, head_dim
        new_batched_prompt = new_batched_prompt.permute(2, 0, 3, 1, 4) # B, dual, length, num_heads, head_dim


        return new_batched_prompt
    
    def conv_ortho(self, weights):
        w = weights
        in_channels, out_channels, kernel_size, kernel_size = w.shape
        w =w.permute(1, 0, 2, 3). view(out_channels, -1)
        W1 = w.t()
        Ident = torch.eye(w.shape[1]).to(w.device)
        # print("W1 shape: ", W1.shape, w.shape)
        W_new = torch.matmul(W1, w)
        Norm = W_new - Ident
        b_k = torch.rand(Norm.shape[1]).to(Norm.device)
        b_k = b_k.unsqueeze(1)
        v1 = torch.matmul(Norm, b_k)
        norm1 = torch.sum(torch.square(v1))**0.5
        v2 = v1 / norm1
        v3 = torch.matmul(Norm, v2)

        return 0.01*(torch.sum(torch.square(v3))**0.5) + (1e-4)*(torch.sum(torch.square(w))**0.5)

    def conv_orthogonality(self, conv_vals):
        ortho_norm = 0
        for i in range(self.num_layers):

            for h in range(self.num_heads):

                conv_val = conv_vals[str(i)][str(h)]
                for j in range(len(conv_val)):
                    ortho_norm += self.conv_ortho(conv_val[j].weight)
        return ortho_norm
