# Báo cáo kỹ thuật: Luồng hoạt động Rainbow Prompt

## 1. Giới thiệu

Rainbow Prompt là cơ chế tiến hóa prompt liên tục dành cho các mô hình Vision Transformer (ViT) trong bối cảnh học liên tục (continual learning). Phương pháp này tìm cách tái sử dụng và tinh chỉnh các prompt đã học từ những tác vụ trước để thích ứng với tác vụ mới, đồng thời kiểm soát độ phức tạp thông qua cơ chế gating xác suất. Hệ thống hỗ trợ hai chế độ tiến hóa:

- `use_paper_evolution: false` – mặc định trong codebase, ưu tiên tính hiệu quả với một bước tổng hợp duy nhất.
- `use_paper_evolution: true` – tuân thủ mô tả trong bài báo, tiến hóa riêng biệt cho từng prompt lịch sử rồi tổng hợp.

## 2. Kiến trúc tổng quan

Pipeline Rainbow Prompt bao gồm bốn thành phần chính:

1. **`RainbowPromptModule`**: quản lý tập prompt nền theo từng layer, thực thi tiến hóa, gating, và lưu trữ.
2. **`RainbowEvolution`**: biến đổi prompt mới dựa trên lịch sử các prompt đã quan sát.
3. **`ProbabilisticGate`**: quyết định mức sử dụng prompt tiến hóa cho từng layer bằng Gumbel-Softmax.
4. **`RainbowPromptStorage`**: lưu lại prompt và trạng thái gate sau khi hoàn tất từng tác vụ.

Các thành phần này được gắn vào backbone ViT thông qua tham số `use_rainbow_prompt`. Trong quá trình forward, mỗi block của ViT có thể nhận thêm token prefix được xây dựng từ Rainbow Prompt trước khi đi vào cơ chế self-attention.

## 3. Quy trình khởi tạo

1. **Nạp cấu hình**: YAML cấu hình cung cấp siêu tham số huấn luyện, bao gồm chiều dài prompt, thông số gating và các hệ số regularization (`lambda_sparse`, `lambda_match`).
2. **Chuẩn bị dữ liệu**: `build_continual_dataloader` tạo ra loader theo từng tác vụ, đồng thời trả về `class_mask` để ánh xạ nhãn.
3. **Khởi tạo mô hình**: `create_model(..., use_rainbow_prompt=True)` tạo ViT với module Rainbow Prompt được gắn vào từng layer attention thích hợp.
4. **Thiết lập matcher**: `RainbowAttributeMatcher` sinh embedding mô tả tác vụ, hỗ trợ điều kiện hóa tiến hóa.
5. **Đặt loss và optimizer**: Cross-entropy là loss chính; optimizer và scheduler lấy từ cấu hình.
6. **Khởi tạo prompt**: `RainbowPromptModule` tạo danh sách `base_prompts` rỗng cho từng layer, thiết lập module tiến hóa và storage, nhưng chưa thêm prompt cụ thể cho tới khi bắt đầu tác vụ đầu tiên.

## 4. Quy trình huấn luyện theo từng tác vụ

Quy trình huấn luyện được thực hiện trong hàm `train_and_evaluate_rainbow()` (file `engine.py`), được gọi từ `main_rainbow()`.

### 4.1 Bắt đầu tác vụ mới (`start_task`)

**Vị trí**: Được gọi tại `engine.py:681` trong vòng lặp `for task_id in range(args.num_tasks)`.

**Module**: `VisionTransformer.rainbow_start_task()` → `RainbowPromptModule.start_task()`

**Đầu vào**:
- `task_id: int` - ID của tác vụ mới (0-indexed)

**Quá trình thực hiện**:
1. **Đánh dấu tác vụ hiện tại**: `self.current_task_id = task_id`
2. **Vô hiệu hóa gradient cho prompt cũ**: Với mỗi layer, duyệt qua tất cả prompt trong `self.base_prompts[layer_idx]` và đặt `prompt.requires_grad = False`
3. **Khởi tạo prompt mới**: 
   - Tạo `nn.Parameter` với shape `[prompt_length, embed_dim]`, khởi tạo bằng `nn.init.normal_(mean=0.0, std=0.02)`
   - Thêm vào `self.base_prompts[layer_idx]` cho mỗi layer
4. **Khởi tạo gate**: Nếu `use_adaptive_gating=True`, tạo `ProbabilisticGate(num_layers, tau_start, tau_end, harden_epoch_ratio)` và gán vào `self.current_gate`
5. **Reset cache**: Xóa `self._latest_layer_cache` và `self._aux_losses`

**Đầu ra**: Không có giá trị trả về, chỉ cập nhật trạng thái nội bộ.

### 4.2 Vòng lặp epoch

**Vị trí**: `engine.py:685-704`, vòng lặp `for epoch in range(args.epochs)` bên trong vòng lặp task.

**Đầu mỗi epoch**:
1. **Cập nhật epoch cho Rainbow Prompt**: Gọi `model.rainbow_set_epoch(epoch, max_epochs)` tại `engine.py:402`
   - Module: `VisionTransformer.rainbow_set_epoch()` → `RainbowPromptModule.set_epoch()`
   - Đầu vào: `epoch: int`, `max_epochs: int`
   - Cập nhật: `self.current_epoch = epoch`, `self.max_epochs = max_epochs`
   - Mục đích: Điều chỉnh nhiệt độ Gumbel-Softmax trong gating theo tiến độ epoch

2. **Khởi tạo metric logger**: Tạo `MetricLogger` để theo dõi loss, accuracy trong epoch

### 4.3 Forward pass cho từng batch

**Vị trí**: `engine.py:416-464`, vòng lặp `for samples, targets in data_loader` trong `train_one_epoch_rainbow()`.

#### Bước 1: Chuẩn bị dữ liệu
- **Đầu vào**: `samples: Tensor[B, C, H, W]`, `targets: Tensor[B]`
- **Xử lý**: 
  - Chuyển sang device: `samples.to(device)`, `targets.to(device)`
  - Tính `offset = _class_offset(class_mask, task_id)` - offset nhãn cho tác vụ hiện tại
  - Tính `task_classes = _task_num_classes(class_mask, task_id)` - số lớp của tác vụ

#### Bước 2: Tạo task embedding
- **Module**: `RainbowAttributeMatcher.get_task_embedding()`
- **Vị trí**: `engine.py:422`
- **Đầu vào**: `task_id: int`, `device: torch.device`
- **Quá trình**:
  - Lấy embedding từ `self.task_embeddings[task_id]` (learnable embedding table)
  - Trả về tensor shape `[embed_dim]`
- **Đầu ra**: `task_embedding: Tensor[embed_dim]`
- **Cập nhật vào model**: `model.rainbow_set_task_embedding(task_embedding)` → `RainbowPromptModule.update_task_embedding()`

#### Bước 3: Forward pass qua model
- **Vị trí**: `engine.py:425`, gọi `model(samples, task_id=task_id, train=True)`
- **Luồng xử lý**:

  **3.1. VisionTransformer.forward()** (`vision_transformer.py:700`)
  - Gọi `forward_features(x, task_id, train=True)` → `forward_head(res)`

  **3.2. VisionTransformer.forward_features()** (`vision_transformer.py:583-667`)
  - **Patch embedding**: `x = self.patch_embed(x)` → `[B, num_patches, embed_dim]`
  - **Thêm CLS token**: `x = torch.cat([cls_token, x], dim=1)` → `[B, num_patches+1, embed_dim]`
  - **Position embedding**: `x = pos_drop(x + pos_embed)`
  - **Vòng lặp qua các block** (`vision_transformer.py:594-601`):
    ```python
    for i, block in enumerate(self.blocks):
        prompt_tokens = self.rainbow_prompt(
            task_id=task_id,
            layer_idx=i,
            batch_size=x.shape[0],
            device=x.device,
        )
        x = block(x, prompt=prompt_tokens)
    ```

  **3.3. RainbowPromptModule.forward()** (`prompt.py:248-266`)
  - **Điều kiện**: `self.training_mode == True` → gọi `_prepare_training_prompt()`
  - **Đầu vào**: `task_id`, `layer_idx`, `batch_size`, `device`
  - **Quá trình**:
    
    a) **Stack prompts lịch sử** (`prompt.py:151`):
       - Gọi `base_prompts = self._stack_prompts(layer_idx)`
       - Lấy tất cả prompt từ `self.base_prompts[layer_idx]` và stack thành tensor `[num_tasks, prompt_length, embed_dim]`
       - Lấy prompt mới nhất: `new_prompt = self.base_prompts[layer_idx][-1]` → `[prompt_length, embed_dim]`
    
    b) **Lấy task embedding**: Nếu `use_task_conditioning=True`, lấy `self.task_embedding` → `[embed_dim]`
    
    c) **Tiến hóa prompt** (`prompt.py:158`):
       - Gọi `evo_out = self.evolutions[layer_idx](base_prompts, new_prompt, task_embedding)`
       - Module: `RainbowEvolution.forward()` (xem chi tiết ở mục 5)
       - Đầu ra: `{"rainbow_prompt": Tensor[prompt_length, embed_dim], ...}`
    
    d) **Gating** (`prompt.py:165-173`):
       - Nếu `use_adaptive_gating=True`:
         - Gọi `gate_out = self.current_gate(layer_idx, epoch, max_epochs, training=True)`
         - Module: `ProbabilisticGate.forward()` (xem chi tiết ở mục 6)
         - Đầu ra: `{"gate": Tensor[], "sparsity_loss": Tensor[], ...}`
         - Lưu `sparsity_loss` vào `self._aux_losses[f"sparsity_{layer_idx}"]`
       - Nếu không: `gate_value = torch.ones(())`
    
    e) **Định dạng prompt** (`prompt.py:177`):
       - Gọi `formatted_prompt = self._format_prompt(rainbow_prompt, gate_value, batch_size)`
       - Chuyển `[prompt_length, embed_dim]` → `[B, 2, prompt_length, num_heads, head_dim]` (prefix format cho attention)
       - Nhân với `gate_value` để điều chỉnh cường độ
       - Đầu ra: `[B, 2, prompt_length, num_heads, head_dim]` (2 = key/value)
    
    f) **Cache kết quả** (`prompt.py:179-182`):
       - Lưu `rainbow_prompt` và `gate_value` vào `self._latest_layer_cache[layer_idx]` để dùng khi `finalize_task()`
  
  **3.4. Block.forward() với prompt** (`vision_transformer.py:253`)
  - **Đầu vào**: `x: [B, seq_len, embed_dim]`, `prompt: [B, 2, prompt_length, num_heads, head_dim]`
  - **Xử lý**: 
    - `x = norm1(x)`
    - `x = attn(x, prompt)` → `PreT_Attention` nhận prompt và chèn vào key/value của attention
    - `x = x + drop_path(ls1(attn_output))`
    - `x = x + drop_path(ls2(mlp(norm2(x))))`
  
  **3.5. Forward head** (`vision_transformer.py:669-698`)
  - Lấy CLS token: `x = x[:, 0]` → `[B, embed_dim]`
  - Lưu vào `res['pre_logits'] = x`
  - Normalize: `x = fc_norm(x)`
  - Classifier: `out = head(x)` → `[B, num_classes_per_task]`
  - Lưu vào `res['logits'] = out['logits']`

#### Bước 4: Thu thập auxiliary losses
- **Vị trí**: `vision_transformer.py:602`
- **Module**: `RainbowPromptModule.auxiliary_losses()`
- **Quá trình**: 
  - Lấy tất cả loss từ `self._aux_losses` (đã được lưu trong forward pass)
  - Xóa cache: `self._aux_losses.clear()`
- **Đầu ra**: `Dict[str, Tensor]` với keys `"sparsity_{layer_idx}"` cho mỗi layer
- **Lưu vào output**: `res["rainbow_aux"] = auxiliary_losses`

### 4.4 Tính toán loss và tối ưu

**Vị trí**: `engine.py:426-450`

#### Bước 1: Tính Cross-Entropy Loss
- **Đầu vào**: 
  - `logits_current = logits[:, offset:offset+task_classes]` → `[B, task_classes]`
  - `adjusted_targets = targets - offset` → `[B]`
- **Module**: `torch.nn.CrossEntropyLoss()`
- **Đầu ra**: `ce_loss: Tensor[]` (scalar)

#### Bước 2: Tính Sparsity Loss
- **Điều kiện**: `args.lambda_sparse > 0` và `aux` không rỗng
- **Vị trí**: `engine.py:437-439`
- **Quá trình**:
  - Lấy `aux = output.get('rainbow_aux', {})`
  - Tổng hợp: `sparsity_loss = sum(aux.values())` (tổng log-probability từ tất cả layers)
- **Đầu ra**: `sparsity_loss: Tensor[]` (scalar)

#### Bước 3: Tính Match Loss
- **Điều kiện**: `args.lambda_match > 0`
- **Vị trí**: `engine.py:441-443`
- **Module**: `RainbowAttributeMatcher.match_loss()`
- **Đầu vào**: 
  - `features = output['pre_logits']` → `[B, embed_dim]`
  - `task_embedding: Tensor[embed_dim]`
- **Quá trình**:
  - Project features: `projected = query_head(features)` → `[B, embed_dim]`
  - Normalize: `projected = F.normalize(projected)`, `target = F.normalize(task_embedding)`
  - Cosine similarity: `cosine = sum(projected * target, dim=-1).mean()`
  - Loss: `match_loss = 1.0 - cosine`
- **Đầu ra**: `match_loss: Tensor[]` (scalar)

#### Bước 4: Tổng hợp và backward
- **Tổng loss**: `total_loss = ce_loss + lambda_sparse * sparsity_loss + lambda_match * match_loss`
- **Backward**: `total_loss.backward()`
- **Gradient clipping**: Nếu `args.clip_grad` được set, `torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)`
- **Optimizer step**: `optimizer.step()`

### 4.5 Kết thúc tác vụ (`finalize_task`)

**Vị trí**: `engine.py:706`, được gọi sau khi hoàn thành tất cả epoch của tác vụ.

**Module**: `VisionTransformer.rainbow_finalize_task()` → `RainbowPromptModule.finalize_task()`

**Đầu vào**: `task_id: int`

**Quá trình**:
1. **Kiểm tra cache**: Nếu `self._latest_layer_cache` rỗng, return ngay
2. **Lưu prompt và gate** (`prompt.py:276-277`):
   - Với mỗi layer trong cache:
     - Lấy `prompt` và `gate` từ `self._latest_layer_cache[layer_idx]`
     - Gọi `self.storage.put(task_id, layer_idx, prompt, gate)`
     - Module: `RainbowPromptStorage.put()` - lưu vào memory cache `self._cache[task_id][layer_idx]`
3. **Lưu ra file** (`prompt.py:278`):
   - Gọi `self.storage.save_task(task_id)`
   - Module: `RainbowPromptStorage.save_task()`
   - Serialize tất cả prompt và gate của task thành dict: `{layer_idx: {"prompt": ..., "gate": ...}}`
   - Lưu vào file: `{save_dir}/task_{task_id:03d}.pt`
4. **Xóa cache**: `self._latest_layer_cache.clear()`

**Đầu ra**: Không có giá trị trả về.

## 5. Chi tiết module RainbowEvolution

**Vị trí**: `rainbow/evolution.py:12-201`

**Module**: `RainbowEvolution`

**Mục đích**: Biến đổi prompt mới dựa trên lịch sử các prompt đã học để tạo ra Rainbow Prompt.

### 5.1 Khởi tạo

**Đầu vào**:
- `embed_dim: int` - chiều embedding (thường là 768 cho ViT-Base)
- `prompt_length: int` - số token trong prompt
- `proj_dim: int` - chiều projection (thường là `embed_dim // 8`)
- `align_hidden_dim: int` - chiều ẩn của mạng alignment
- `num_heads: int` - số attention heads
- `use_task_conditioning: bool` - có sử dụng task embedding không
- `enable_task_level: bool` - có bật attention mức task không
- `enable_feature_level: bool` - có bật attention mức feature không
- `enable_alignment: bool` - có bật mạng alignment không
- `use_paper_evolution: bool` - chế độ tiến hóa (false = gộp đơn bước, true = tiến hóa riêng)

**Các module con được tạo**:
- `task_proj: Linear(embed_dim, proj_dim)` - nếu `use_task_conditioning=True`
- `query_proj: Linear(embed_dim, proj_dim)` - projection cho query (prompt mới)
- `key_proj: Linear(embed_dim, proj_dim)` - projection cho key (prompt lịch sử)
- `value_proj: Linear(embed_dim, proj_dim)` - projection cho value (prompt lịch sử)
- `output_proj: Linear(proj_dim, embed_dim)` - projection ngược về embed_dim
- `alignment: Sequential(Linear(embed_dim, align_hidden_dim), ReLU, Linear(align_hidden_dim, embed_dim))` - nếu `enable_alignment=True`
- `layer_norm_in: LayerNorm(embed_dim)`
- `layer_norm_out: LayerNorm(embed_dim)`

### 5.2 Forward pass - Chế độ mặc định (`use_paper_evolution: false`)

**Vị trí**: `rainbow/evolution.py:55-180`

**Đầu vào**:
- `base_prompts: Tensor[num_tasks, prompt_length, embed_dim]` - stack của tất cả prompt lịch sử
- `new_prompt: Tensor[prompt_length, embed_dim]` - prompt mới của task hiện tại
- `task_embedding: Optional[Tensor[embed_dim]]` - embedding mô tả task (nếu có)

**Quá trình**:

1. **Khởi tạo trọng số** (`evolution.py:100`):
   - `task_weights = ones(num_tasks) / num_tasks` → `[num_tasks]` (uniform ban đầu)

2. **Task conditioning** (`evolution.py:102-125`):
   - Nếu `use_task_conditioning=True` và `task_embedding` không None:
     - **Theo paper (Section 3.2.1)**: Task conditioning sử dụng attention-based combination
     - Project task embedding: `task_vec = sigmoid(task_proj(task_embedding))` → `[proj_dim]` (sử dụng sigmoid như trong paper)
     - Project base prompts: `projected_prompts = key_proj(base_prompts)` → `[num_tasks, prompt_length, proj_dim]`
     - Tính similarity per prompt: `key_repr = projected_prompts.mean(dim=1)` → `[num_tasks, proj_dim]` (pool theo prompt_length)
     - Tính scores: `scores = matmul(key_repr, task_vec) / sqrt(proj_dim)` → `[num_tasks]`
     - Softmax: `task_weights_conditioning = softmax(scores, dim=0)` → `[num_tasks]`
     - **Attention-based combination** (theo paper): `conditioned_single = einsum("p,pld->ld", task_weights_conditioning, base_prompts)` → `[prompt_length, embed_dim]`
       - Tạo một prompt đại diện từ tất cả base prompts thông qua weighted sum
     - Expand để giữ shape: `conditioned_prompts = conditioned_single.unsqueeze(0).expand(num_tasks, -1, -1)` → `[num_tasks, prompt_length, embed_dim]`
   - Nếu không: `conditioned_prompts = base_prompts`

3. **Project prompts** (`evolution.py:123-126`):
   - `query = query_proj(new_prompt)` → `[prompt_length, proj_dim]`
   - `keys = key_proj(conditioned_prompts)` → `[num_tasks, prompt_length, proj_dim]`
   - `values = value_proj(conditioned_prompts)` → `[num_tasks, prompt_length, proj_dim]`

4. **Task-level attention** (`evolution.py:130-141`):
   - Nếu `enable_task_level=True`:
     - Pool query: `pooled_query = query.mean(dim=0, keepdim=True)` → `[1, proj_dim]`
     - Pool keys: `pooled_keys = keys.mean(dim=1).transpose(0, 1)` → `[proj_dim, num_tasks]`
     - Tính attention: `logits = matmul(pooled_query, pooled_keys) / sqrt(proj_dim)` → `[1, num_tasks]`
     - Softmax: `task_attn = softmax(logits.squeeze(0))` → `[num_tasks]`
     - Kết hợp với task conditioning:
       - Nếu có task conditioning: `task_weights = 0.5 * task_weights + 0.5 * task_attn`
         - Lưu ý: `task_weights` ban đầu được khởi tạo uniform, sau đó được kết hợp với `task_attn`
         - Task conditioning weights (`task_weights_conditioning`) đã được sử dụng để tạo `conditioned_prompts`
       - Nếu không: `task_weights = task_attn`
   - Normalize: `task_weights = task_weights / task_weights.sum().clamp(min=1e-6)`

5. **Gộp đơn bước** (`evolution.py:143-146`):
   - Weighted values: `weighted_values = einsum("p,pld->ld", task_weights, values)` → `[prompt_length, proj_dim]`
   - Weighted keys: `weighted_keys = einsum("p,pld->ld", task_weights, keys)` → `[prompt_length, proj_dim]`
   - Weighted prompts: `weighted_prompts = einsum("p,pld->ld", task_weights, conditioned_prompts)` → `[prompt_length, embed_dim]`

6. **Feature-level attention** (`evolution.py:152-163`):
   - Nếu `enable_feature_level=True`:
     - **Theo paper**: Feature-level attention sử dụng `Q^T @ K` (attention theo chiều feature dimension)
     - Transpose query và keys: `query.transpose(-1, -2)` → `[proj_dim, prompt_length]`, `weighted_keys` → `[prompt_length, proj_dim]`
     - Attention logits: `feature_logits = matmul(query.transpose(-1, -2), weighted_keys) / sqrt(proj_dim)` → `[proj_dim, proj_dim]`
       - Đây là attention giữa các feature dimensions, không phải giữa các positions
     - Softmax: `feature_attn = softmax(feature_logits, dim=-1)` → `[proj_dim, proj_dim]`
     - Apply attention to values: `evolved_proj = matmul(feature_attn, weighted_values.transpose(-1, -2)).transpose(-1, -2)` → `[prompt_length, proj_dim]`
       - Transpose values: `weighted_values.transpose(-1, -2)` → `[proj_dim, prompt_length]`
       - Apply: `feature_attn @ values^T` → `[proj_dim, prompt_length]`
       - Transpose back: `[prompt_length, proj_dim]`
   - Nếu không: `evolved_proj = weighted_values`

7. **Output projection và layer norm** (`evolution.py:164-165`):
   - Project: `evolved_embeds = output_proj(evolved_proj)` → `[prompt_length, embed_dim]`
   - Residual + norm: `evolved_embeds = layer_norm_in(weighted_prompts + evolved_embeds)` → `[prompt_length, embed_dim]`

8. **Alignment** (`evolution.py:166-170`):
   - Nếu `enable_alignment=True`:
     - `aligned = layer_norm_out(evolved_embeds + alignment(evolved_embeds))` → `[prompt_length, embed_dim]`
   - Nếu không: `aligned = layer_norm_out(evolved_embeds)`

9. **Nhân bản cho logging** (`evolution.py:172`):
   - `aligned_prompts = stack([aligned for _ in range(num_tasks)], dim=0)` → `[num_tasks, prompt_length, embed_dim]`

**Đầu ra**: Dict với keys:
- `"rainbow_prompt": Tensor[prompt_length, embed_dim]` - prompt đã tiến hóa
- `"aligned_prompts": Tensor[num_tasks, prompt_length, embed_dim]` - nhân bản cho logging
- `"task_weights": Tensor[num_tasks]` - trọng số đã sử dụng (detached)
- `"feature_attn": Optional[Tensor[proj_dim, proj_dim]]` - attention weights mức feature (nếu có, detached)
  - **Lưu ý**: Shape đã thay đổi từ `[prompt_length, prompt_length]` (position-level) sang `[proj_dim, proj_dim]` (feature-level) để đúng với paper

### 5.3 Forward pass - Chế độ theo bài báo (`use_paper_evolution: true`)

**Vị trí**: `rainbow/evolution.py:180-230`

**Đầu vào**: Giống chế độ mặc định.

**Quá trình**:

1. **Task conditioning và task-level attention**: Giống chế độ mặc định (bước 1-4).

2. **Nhân trọng số cho từng prompt** (`evolution.py:180-183`):
   - `expanded_weights = task_weights.view(num_tasks, 1, 1)` → `[num_tasks, 1, 1]`
   - `weighted_values = expanded_weights * values` → `[num_tasks, prompt_length, proj_dim]`
   - `weighted_keys = expanded_weights * keys` → `[num_tasks, prompt_length, proj_dim]`

3. **Tiến hóa riêng cho từng prompt** (`evolution.py:187-210`):
   - Vòng lặp: `for prompt_idx in range(num_tasks)`:
     - Lấy value và key của prompt i: `value_i = weighted_values[prompt_idx]`, `key_i = weighted_keys[prompt_idx]`
     - Lấy prompt gốc: `prompt_i = conditioned_prompts[prompt_idx]`
     - **Feature-level attention** (nếu bật):
       - **Theo paper**: Feature-level attention sử dụng `Q^T @ K` (attention theo chiều feature dimension)
       - Transpose query và key: `query.transpose(-1, -2)` → `[proj_dim, prompt_length]`, `key_i` → `[prompt_length, proj_dim]`
       - Attention logits: `logits = matmul(query.transpose(-1, -2), key_i) / sqrt(proj_dim)` → `[proj_dim, proj_dim]`
       - Softmax: `attn = softmax(logits, dim=-1)` → `[proj_dim, proj_dim]`
       - Apply attention to values: `evolved_proj = matmul(attn, value_i.transpose(-1, -2)).transpose(-1, -2)` → `[prompt_length, proj_dim]`
         - Transpose value: `value_i.transpose(-1, -2)` → `[proj_dim, prompt_length]`
         - Apply: `attn @ value_i^T` → `[proj_dim, prompt_length]`
         - Transpose back: `[prompt_length, proj_dim]`
     - Nếu không bật: `evolved_proj = value_i`
     - **Output projection và layer norm**:
       - `evolved_embeds = output_proj(evolved_proj)` → `[prompt_length, embed_dim]`
       - `evolved_embeds = layer_norm_in(prompt_i + evolved_embeds)` → `[prompt_length, embed_dim]`
     - **Alignment**:
       - Nếu bật: `aligned_i = layer_norm_out(evolved_embeds + alignment(evolved_embeds))`
       - Nếu không: `aligned_i = layer_norm_out(evolved_embeds)`
     - Lưu vào list: `aligned_prompts_list.append(aligned_i)`

4. **Tổng hợp** (`evolution.py:212-214`):
   - Stack: `aligned_prompts = stack(aligned_prompts_list, dim=0)` → `[num_tasks, prompt_length, embed_dim]`
   - Trung bình: `rainbow_prompt = aligned_prompts.mean(dim=0)` → `[prompt_length, embed_dim]`

**Đầu ra**: Dict với keys giống chế độ mặc định.

**So sánh hai chế độ**:
- **Chế độ mặc định**: Gộp trước rồi mới attention → nhanh hơn, mất một số thông tin vi mô
- **Chế độ bài báo**: Attention riêng cho từng prompt rồi mới gộp → chậm hơn, giữ được nhiều thông tin hơn

**Lưu ý về các thay đổi theo paper**:
- **Task Conditioning**: Đã được cập nhật từ element-wise scaling sang attention-based combination (weighted sum) theo Section 3.2.1 của paper. Sử dụng `sigmoid` thay vì `tanh` cho task embedding.
- **Feature-level Attention**: Đã được cập nhật từ position-level attention (`Q @ K^T` → `[prompt_length, prompt_length]`) sang feature-level attention (`Q^T @ K` → `[proj_dim, proj_dim]`) để đúng với mô tả trong paper về "denoising noise dimensions".

## 6. Chi tiết module ProbabilisticGate

**Vị trí**: `rainbow/gating.py:12-74`

**Module**: `ProbabilisticGate`

**Mục đích**: Quyết định mức độ sử dụng Rainbow Prompt cho từng layer thông qua phân phối xác suất, với cơ chế Gumbel-Softmax để có thể học được.

### 6.1 Khởi tạo

**Đầu vào**:
- `num_layers: int` - số layer trong ViT
- `tau_start: float` - nhiệt độ ban đầu (thường 1.0)
- `tau_end: float` - nhiệt độ cuối (thường 0.3)
- `harden_epoch_ratio: float` - tỷ lệ epoch để chuyển sang hard selection (thường 0.6)

**Tham số được tạo**:
- `self.logits: Parameter[num_layers, 2]` - logits cho mỗi layer, mỗi layer có 2 trạng thái: [tắt, bật]
- Khởi tạo: `torch.zeros(num_layers, 2)`

### 6.2 Forward pass

**Vị trí**: `rainbow/gating.py:30-73`

**Đầu vào**:
- `layer_idx: int` - chỉ số layer (0-indexed)
- `epoch: int` - epoch hiện tại
- `max_epochs: int` - tổng số epoch
- `training: bool` - có đang training không
- `temperature: Optional[float]` - nhiệt độ tùy chỉnh (nếu None thì tính tự động)

**Quá trình**:

1. **Lấy logits cho layer** (`gating.py:41`):
   - `logits = self.logits[layer_idx]` → `[2]` (logits cho trạng thái tắt và bật)

2. **Tính nhiệt độ** (`gating.py:42-46`):
   - Nếu `temperature` được cung cấp: `tau = temperature`
   - Nếu không:
     - Tính progress: `progress = min(max(epoch, 0) / max(max_epochs, 1), 1.0)` → [0, 1]
     - Nội suy tuyến tính: `tau = tau_start + (tau_end - tau_start) * progress`
     - Ví dụ: nếu `tau_start=1.0`, `tau_end=0.3`, `epoch=5`, `max_epochs=10`:
       - `progress = 5/10 = 0.5`
       - `tau = 1.0 + (0.3 - 1.0) * 0.5 = 0.65`

3. **Xác định có dùng hard selection không** (`gating.py:48`):
   - `use_hard = training and progress >= harden_epoch_ratio`
   - Ví dụ: nếu `harden_epoch_ratio=0.6`, `max_epochs=10`:
     - Từ epoch 6 trở đi (`progress >= 0.6`), sử dụng hard selection

4. **Tính gate value**:

   **a) Trong training** (`gating.py:50-56`):
   - **Gumbel noise**: `gumbel_noise = -log(-log(rand_like(logits)))` → `[2]`
   - **Gumbel-Softmax**: `y = softmax((logits + gumbel_noise) / tau, dim=-1)` → `[2]`
   - **Hard selection** (nếu `use_hard=True`):
     - `index = argmax(y, dim=-1)` → scalar (0 hoặc 1)
     - `y_hard = one_hot(index, num_classes=2).float()` → `[2]` (one-hot)
     - **Straight-through estimator**: `y = (y_hard - y).detach() + y`
       - Trong forward: sử dụng `y_hard` (discrete)
       - Trong backward: gradient đi qua `y` (continuous)
   
   **b) Trong inference** (`gating.py:57-60`):
   - Tính xác suất: `probs_eval = softmax(logits, dim=-1)` → `[2]`
   - Chọn trạng thái có xác suất cao nhất: `index = argmax(probs_eval, dim=-1)`
   - One-hot: `y = one_hot(index, num_classes=2).float()` → `[2]`

5. **Tính sparsity loss** (`gating.py:62-66`):
   - Xác suất mềm: `probs_soft = softmax(logits, dim=-1)` → `[2]`
   - Xác suất sử dụng: `prob_use = probs_soft[1]` → scalar (xác suất trạng thái "bật")
   - Gate value: `gate_value = y[1]` → scalar (0 hoặc 1 trong hard mode, [0, 1] trong soft mode)
   - **Sparsity loss**: `sparsity_loss = log(prob_use + 1e-8)`
     - Nếu `prob_use` gần 1 → `sparsity_loss` gần 0 (không penalty)
     - Nếu `prob_use` gần 0 → `sparsity_loss` rất âm (penalty lớn, khuyến khích tắt)

**Đầu ra**: Dict với keys:
- `"gate": Tensor[]` - giá trị gate (scalar, shape `[]`)
- `"prob_use": Tensor[]` - xác suất sử dụng (scalar)
- `"logits": Tensor[2]` - logits gốc
- `"sparsity_loss": Tensor[]` - loss cho regularization

**Ý nghĩa**:
- Gate value được nhân với prompt trong `_format_prompt()` để điều chỉnh cường độ
- Sparsity loss được tổng hợp từ tất cả layers và nhân với `lambda_sparse` trong total loss
- Cơ chế Gumbel-Softmax cho phép học được trong khi vẫn có thể chuyển sang discrete selection

## 7. Quy trình suy luận và đánh giá

### 7.1 Đánh giá sau mỗi tác vụ (`evaluate_rainbow_till_now`)

**Vị trí**: `engine.py:589-663`, được gọi tại `engine.py:708` sau khi `finalize_task()`.

**Module**: `evaluate_rainbow_till_now()`

**Đầu vào**:
- `model: VisionTransformer`
- `matcher: RainbowAttributeMatcher`
- `data_loader: Dict[int, Dict[str, DataLoader]]` - dict chứa data loader cho từng task
- `device: torch.device`
- `task_id: int` - task vừa hoàn thành
- `class_mask: List[List[int]]`
- `acc_matrix: np.ndarray` - ma trận accuracy [num_tasks, num_tasks]
- `args`

**Quá trình**:

1. **Nạp prompt cho tất cả tác vụ đã học** (`engine.py:609`):
   - Gọi `model.rainbow_load_all_tasks(task_id, device)`
   - Module: `VisionTransformer.rainbow_load_all_tasks()` → `RainbowPromptModule.load_task()` cho mỗi task
   - Mục đích: Nạp tất cả prompt từ storage vào memory cache để có thể sử dụng trong inference

2. **Đánh giá tuần tự trên từng tác vụ** (`engine.py:611-620`):
   - Vòng lặp: `for eval_task in range(task_id + 1)`
   - Gọi `evaluate_rainbow()` cho từng task (xem chi tiết ở 7.2)
   - Cập nhật `acc_matrix[eval_task, task_id]` với accuracy của task `eval_task` sau khi học task `task_id`

3. **Tính toán metrics tổng hợp** (`engine.py:630-644`):
   - **Average accuracy**: Trung bình accuracy của tất cả task đã học
   - **Forgetting**: Nếu `task_id > 0`:
     - `previous_max = max(acc_matrix[:, :task_id], axis=1)` - accuracy tốt nhất trước đó của mỗi task
     - `current_acc = acc_matrix[:, task_id]` - accuracy hiện tại
     - `forgetting = mean((previous_max - current_acc)[:task_id])` - mức độ quên
   - **Backward transfer**: 
     - `diagonal = diag(acc_matrix)` - accuracy tại thời điểm học task đó
     - `backward = mean((current_acc - diagonal)[:task_id])` - cải thiện của task cũ nhờ học task mới

**Đầu ra**: Dict với keys `'avg_acc1'`, `'avg_acc5'`, `'avg_loss'`, `'forgetting'`, `'backward'`, `'summary'`

### 7.2 Đánh giá một tác vụ (`evaluate_rainbow`)

**Vị trí**: `engine.py:468-586`

**Module**: `evaluate_rainbow()`

**Đầu vào**:
- `model: VisionTransformer`
- `matcher: RainbowAttributeMatcher`
- `data_loader: DataLoader` - loader cho một task cụ thể
- `device: torch.device`
- `task_id: int` - task cần đánh giá
- `class_mask: List[List[int]]`
- `args`

**Quá trình**:

1. **Thiết lập chế độ evaluation** (`engine.py:477-478`):
   - `model.eval()` - đặt model ở chế độ evaluation
   - `model.rainbow_prompt.set_training(False)` - đặt Rainbow Prompt ở chế độ inference

2. **Tính offset và số lớp** (`engine.py:484-485`):
   - `offset = _class_offset(class_mask, task_id)`
   - `task_classes = _task_num_classes(class_mask, task_id)`

3. **Vòng lặp qua batch** (`engine.py:505-554`):

   **3.1. Chuẩn bị dữ liệu**:
   - `samples.to(device)`, `targets.to(device)`

   **3.2. Chọn chế độ routing prompt**:
   
   **a) Hard prompt selection** (`engine.py:515-534`):
   - Điều kiện: `model.rainbow_prompt.hard_prompt_selection == True`
   - **Bước 1 - Soft pass để dự đoán task**:
     - Tạm thời đặt `hard_prompt_selection = False`
     - Forward: `tmp_out = model(samples, task_id=task_id, train=False)`
     - Lấy `pre_logits = tmp_out['pre_logits']` → `[B, embed_dim]`
     - Dự đoán task: `pred_task_id = matcher.predict_task_id(pre_logits, device)`
       - Module: `RainbowAttributeMatcher.predict_task_id()`
       - Quá trình:
         - Aggregate batch: `query = features.mean(dim=0)` → `[1, embed_dim]`
         - Normalize: `query = F.normalize(query)`
         - Lấy tất cả task embeddings: `task_embs = self.task_embeddings.weight` → `[num_tasks, embed_dim]`
         - Normalize: `task_embs = F.normalize(task_embs, dim=-1)`
         - Cosine similarity: `sims = matmul(query, task_embs.t())` → `[num_tasks]`
         - Chọn task có similarity cao nhất: `pred_task_id = argmax(sims)`
     - Khôi phục flag: `hard_prompt_selection = prev_flag`
     - Cập nhật thống kê: `task_match_total += 1`, nếu `pred_task_id == task_id` thì `task_match_correct += 1`
   
   - **Bước 2 - Hard pass với task đã dự đoán**:
     - Lấy task embedding: `task_embedding = matcher.get_task_embedding(pred_task_id, device)`
     - Cập nhật: `model.rainbow_set_task_embedding(task_embedding)`
     - Forward: `output = model(samples, task_id=pred_task_id, train=False)`
   
   **b) Soft prompt selection (mặc định)** (`engine.py:536-539`):
   - Lấy task embedding: `task_embedding = matcher.get_task_embedding(task_id, device)`
   - Cập nhật: `model.rainbow_set_task_embedding(task_embedding)`
   - Forward: `output = model(samples, task_id=task_id, train=False)`

   **3.3. Forward pass với Rainbow Prompt ở chế độ inference**:
   
   **a) VisionTransformer.forward()** → `forward_features()`:
   - Tương tự training, nhưng `train=False`
   
   **b) RainbowPromptModule.forward()** (`prompt.py:248-266`):
   - **Điều kiện**: `self.training_mode == False`
   - **Chọn phương thức**:
     - Nếu `hard_prompt_selection == True`: gọi `_prepare_single_task_inference_prompt(task_id, layer_idx, batch_size, device)`
     - Nếu `hard_prompt_selection == False`: gọi `_prepare_inference_prompt(layer_idx, batch_size, device)`
   
   **c) _prepare_single_task_inference_prompt()** (`prompt.py:227-246`):
   - **Mục đích**: Sử dụng prompt của một task cụ thể (hard routing)
   - **Quá trình**:
     - Lấy từ storage: `stored = self.storage.get(task_id, layer_idx)`
       - Module: `RainbowPromptStorage.get()`
       - Nếu chưa có trong cache, load từ file `task_{task_id:03d}.pt`
       - Trả về `{"prompt": Tensor[prompt_length, embed_dim], "gate": Tensor[]}`
     - Lấy prompt và gate: `prompt_tensor = stored["prompt"]`, `gate_value = stored["gate"]`
     - Format: `formatted_prompt = self._format_prompt(prompt_tensor, gate_value, batch_size)`
     - Đầu ra: `[B, 2, prompt_length, num_heads, head_dim]`
   
   **d) _prepare_inference_prompt()** (`prompt.py:186-225`):
   - **Mục đích**: Ghép prompt từ tất cả task đã học (soft routing)
   - **Quá trình**:
     - **Lazy load prompts** (`prompt.py:195-207`):
       - Nếu cache rỗng, quét thư mục `save_dir` tìm file `task_*.pt`
       - Load từng file vào cache: `self.storage.load_task(task_idx, device)`
     - **Thu thập prompts** (`prompt.py:209-216`):
       - Duyệt qua `self.storage._cache` theo thứ tự task_id tăng dần
       - Lấy prompt của layer hiện tại: `prompts.append(data["prompt"].to(device))`
       - Mỗi prompt có shape `[prompt_length, embed_dim]`
     - **Ghép prompts** (`prompt.py:222-223`):
       - Concatenate: `prompt_tensor = torch.cat(prompts, dim=0)` → `[T*prompt_length, embed_dim]`
       - T là số task đã học
     - **Format**: `formatted_prompt = self._format_prompt(prompt_tensor, gate_value=1.0, batch_size)`
       - Shape: `[B, 2, T*prompt_length, num_heads, head_dim]`
     - **Cơ chế soft routing**: ViT attention sẽ tự động chọn prompt phù hợp thông qua self-attention weights

   **3.4. Tính toán metrics**:
   - Lấy logits: `logits_current = logits[:, offset:offset+task_classes]`
   - Tính loss: `loss = CrossEntropyLoss(logits_current, adjusted_targets)`
   - Tính accuracy: `acc1, acc5 = accuracy(logits_current, adjusted_targets, topk=(1, 5))`
   - Cập nhật thống kê: `total += batch_size`, `correct_top1 += acc1/100 * batch_size`, `correct_top5 += acc5/100 * batch_size`

4. **Tổng hợp và báo cáo** (`engine.py:556-586`):
   - Tính average accuracy: `avg_acc1 = metric_logger.meters['Acc@1'].global_avg`
   - Tính task routing accuracy (nếu hard selection): `task_acc = 100.0 * task_match_correct / task_match_total`
   - In và log kết quả

**Đầu ra**: Dict với keys `'loss'`, `'acc1'`, `'acc5'`, `'total'`, `'correct_top1'`, `'correct_top5'`, `'task_route_acc'`

## 7.3 Chi tiết module RainbowPromptStorage

**Vị trí**: `rainbow/storage.py:11-74`

**Module**: `RainbowPromptStorage`

**Mục đích**: Quản lý việc lưu trữ và truy xuất prompt và gate cho từng task và layer.

### 7.3.1 Khởi tạo

**Đầu vào**:
- `root: str | Path` - đường dẫn thư mục lưu trữ

**Tham số được tạo**:
- `self.root: Path` - thư mục lưu trữ (tự động tạo nếu chưa có)
- `self._cache: Dict[int, Dict[int, Dict[str, Tensor]]]` - cache trong memory
  - Cấu trúc: `{task_id: {layer_idx: {"prompt": Tensor, "gate": Tensor}}}`

### 7.3.2 Các phương thức chính

#### `put(task_id, layer_idx, prompt, gate)`

**Vị trí**: `rainbow/storage.py:19-25`

**Mục đích**: Lưu prompt và gate vào memory cache.

**Đầu vào**:
- `task_id: int`
- `layer_idx: int`
- `prompt: Tensor[prompt_length, embed_dim]`
- `gate: Tensor[]` (scalar)

**Quá trình**:
- Tạo dict cho task nếu chưa có: `self._cache[task_id] = {}`
- Lưu vào cache: `self._cache[task_id][layer_idx] = {"prompt": prompt.detach().cpu().clone(), "gate": gate.detach().cpu().clone()}`

**Đầu ra**: Không có.

#### `get(task_id, layer_idx)`

**Vị trí**: `rainbow/storage.py:27-42`

**Mục đích**: Lấy prompt và gate từ cache hoặc file.

**Đầu vào**:
- `task_id: int`
- `layer_idx: int`

**Quá trình**:
1. **Kiểm tra cache** (`storage.py:28-34`):
   - Nếu `task_id` chưa có trong cache:
     - Tạo đường dẫn file: `file_path = root / f"task_{task_id:03d}.pt"`
     - Nếu file tồn tại: load vào cache
       - `data = torch.load(file_path, map_location="cpu")`
       - Parse: `self._cache[task_id] = {int(k): {"prompt": v["prompt"], "gate": v["gate"]} for k, v in data.items()}`
     - Nếu file không tồn tại: return `None`
2. **Lấy từ cache** (`storage.py:36-42`):
   - `stored = self._cache[task_id].get(layer_idx)`
   - Nếu không có: return `None`
   - Nếu có: return `{"prompt": stored["prompt"].clone(), "gate": stored["gate"].clone()}`

**Đầu ra**: `Optional[Dict[str, Tensor]]` với keys `"prompt"` và `"gate"`, hoặc `None` nếu không tìm thấy.

#### `save_task(task_id)`

**Vị trí**: `rainbow/storage.py:44-53`

**Mục đích**: Lưu tất cả prompt và gate của một task ra file.

**Đầu vào**: `task_id: int`

**Quá trình**:
1. Kiểm tra: nếu `task_id` không có trong cache, raise `KeyError`
2. Serialize (`storage.py:48-51`):
   - Tạo dict: `{layer_idx: {"prompt": prompt.cpu(), "gate": gate.cpu()} for layer_idx, data in cache[task_id].items()}`
3. Lưu file (`storage.py:52-53`):
   - `file_path = root / f"task_{task_id:03d}.pt"`
   - `torch.save(serialized, file_path)`

**Đầu ra**: Không có.

#### `load_task(task_id, device)`

**Vị trí**: `rainbow/storage.py:55-67`

**Mục đích**: Nạp prompt và gate của một task từ file vào cache.

**Đầu vào**:
- `task_id: int`
- `device: Optional[torch.device]` - device để chuyển tensor (None = CPU)

**Quá trình**:
1. Tạo đường dẫn: `file_path = root / f"task_{task_id:03d}.pt"`
2. Kiểm tra file: nếu không tồn tại, raise `FileNotFoundError`
3. Load file (`storage.py:60-67`):
   - `data = torch.load(file_path, map_location=device or "cpu")`
   - Parse và chuyển device: `self._cache[task_id] = {int(k): {"prompt": v["prompt"].to(device), "gate": v["gate"].to(device)} for k, v in data.items()}`

**Đầu ra**: Không có.

**Lưu ý**: Phương thức này được gọi trong:
- `RainbowPromptModule.load_task()` - nạp một task cụ thể
- `RainbowPromptModule._prepare_inference_prompt()` - lazy load khi cần (nếu cache rỗng)
- `VisionTransformer.rainbow_load_all_tasks()` - nạp tất cả task đã học

## 8. So sánh hai chế độ tiến hóa

| Thuộc tính | `use_paper_evolution: false` | `use_paper_evolution: true` |
|------------|-------------------------------|-----------------------------|
| Chi phí tính toán | Thấp (một lần gộp) | Cao (lặp trên từng prompt lịch sử) |
| Bảo toàn thông tin lịch sử | Gián tiếp thông qua gộp | Trực tiếp, giữ cấu trúc từng prompt |
| Độ linh hoạt | Phù hợp với số lượng prompt lớn | Phù hợp khi cần độ chính xác cao |
| Hành vi gating | Giống nhau (gating nằm ngoài tiến hóa) | Giống nhau |

Khuyến nghị: bắt đầu với chế độ mặc định để đạt cân bằng hiệu năng – chi phí; chuyển sang chế độ bài báo khi dữ liệu đòi hỏi tiến hóa tinh vi hoặc để so sánh với kết quả công bố.

## 9. Tóm tắt các module và chức năng

| Module | File | Chức năng chính | Đầu vào chính | Đầu ra chính | Được gọi từ |
|--------|------|-----------------|---------------|-------------|-------------|
| **RainbowPromptModule** | `prompt.py:14` | Quản lý prompt, tiến hóa, gating, storage | `task_id`, `layer_idx`, `batch_size`, `device` | `formatted_prompt: [B, 2, L, H, D]` | `VisionTransformer.forward_features()` |
| **RainbowEvolution** | `rainbow/evolution.py:12` | Tiến hóa prompt dựa trên lịch sử | `base_prompts: [T, L, D]`, `new_prompt: [L, D]`, `task_embedding: [D]` | `{"rainbow_prompt": [L, D], ...}` | `RainbowPromptModule._prepare_training_prompt()` |
| **ProbabilisticGate** | `rainbow/gating.py:12` | Quyết định mức sử dụng prompt qua Gumbel-Softmax | `layer_idx`, `epoch`, `max_epochs`, `training` | `{"gate": scalar, "sparsity_loss": scalar, ...}` | `RainbowPromptModule._prepare_training_prompt()` |
| **RainbowPromptStorage** | `rainbow/storage.py:11` | Lưu trữ và truy xuất prompt/gate | `task_id`, `layer_idx`, `prompt`, `gate` | `{"prompt": [L, D], "gate": scalar}` | `RainbowPromptModule.finalize_task()`, `load_task()` |
| **RainbowAttributeMatcher** | `attribute_matching.py:162` | Tạo task embedding và tính match loss | `task_id`, `features: [B, D]` | `task_embedding: [D]`, `match_loss: scalar` | `train_one_epoch_rainbow()`, `evaluate_rainbow()` |
| **VisionTransformer** | `vision_transformer.py:328` | Backbone ViT với tích hợp Rainbow Prompt | `x: [B, C, H, W]`, `task_id`, `train` | `{"logits": [B, num_classes], "pre_logits": [B, D], "rainbow_aux": {...}}` | `train_one_epoch_rainbow()`, `evaluate_rainbow()` |

**Ký hiệu**:
- `B`: batch size
- `T`: số task đã học
- `L`: prompt_length
- `D`: embed_dim
- `H`: num_heads
- `C`: số channels input
- `H, W`: chiều cao, rộng ảnh

## 10. Kết luận

Rainbow Prompt cung cấp cách tiếp cận có hệ thống để khai thác prompt trong học liên tục với các đặc điểm chính:

### 10.1 Kiến trúc

- **Khởi tạo linh hoạt**: Mỗi task mới được khởi tạo prompt riêng, trong khi prompt cũ được bảo toàn và tái sử dụng thông qua tiến hóa có điều kiện.
- **Tiến hóa đa cấp**: Kết hợp attention mức task và mức feature để tạo ra prompt phù hợp với tác vụ mới dựa trên lịch sử.
- **Gating xác suất**: Tự động điều chỉnh mức độ sử dụng prompt cho từng layer thông qua Gumbel-Softmax, giảm thiểu chi phí tính toán không cần thiết.
- **Storage hiệu quả**: Lưu trữ prompt và gate theo từng task, cho phép đánh giá và suy luận chính xác trên tất cả tác vụ đã học.

### 10.2 Quy trình hoạt động

**Training**:
1. Khởi tạo prompt mới cho task hiện tại
2. Tiến hóa prompt dựa trên lịch sử và task embedding
3. Áp dụng gating để điều chỉnh cường độ
4. Tính loss (CE + sparsity + match) và cập nhật tham số
5. Lưu prompt và gate sau khi hoàn thành task

**Evaluation**:
1. Nạp prompt từ storage cho tất cả task đã học
2. Chọn chế độ routing (hard/soft) dựa trên cấu hình
3. Forward pass với prompt đã nạp
4. Tính accuracy và các metrics (forgetting, backward transfer)

### 10.3 Tùy chọn cấu hình

- **Hai chế độ tiến hóa**: Mặc định (gộp đơn bước) cho tốc độ, hoặc theo bài báo (tiến hóa riêng) cho độ chính xác.
- **Hard/Soft routing**: Hard routing chọn một task cụ thể, soft routing ghép tất cả prompt và để attention tự chọn.
- **Task conditioning**: Có thể bật/tắt việc sử dụng task embedding để điều kiện hóa tiến hóa.

Hai chế độ tiến hóa và các tùy chọn routing cho phép cân đối giữa chi phí tính toán và độ trung thực với mô tả gốc, giúp phương pháp thích nghi với nhiều yêu cầu triển khai khác nhau.



