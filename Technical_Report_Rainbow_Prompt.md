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

### 4.1 Bắt đầu tác vụ mới (`start_task`)

- Đánh dấu `current_task_id` và chuyển tất cả prompt cũ sang trạng thái không cập nhật gradient.
- Khởi tạo prompt mới cho mỗi layer bằng phân phối Gaussian chuẩn.
- Tạo `ProbabilisticGate` (nếu bật `use_adaptive_gating`) và reset cache auxiliary loss.

### 4.2 Vòng lặp epoch

Đối với mỗi tác vụ, huấn luyện kéo dài `args.epochs` epoch. Đầu mỗi epoch:
- Cập nhật `epoch` và `max_epochs` cho Rainbow Prompt để điều chỉnh nhiệt độ gating.
- Khởi tạo lại thống kê logger cho vòng lặp batch.

### 4.3 Forward pass cho từng batch

1. **Chuẩn hóa dữ liệu**: chuyển samples/labels sang `device`, tính offset lớp con tương ứng với tác vụ hiện tại.
2. **Task embedding**: `RainbowAttributeMatcher` sinh vector đặc trưng tác vụ rồi truyền cho Rainbow Prompt.
3. **Tích hợp vào ViT**:
   - `forward_features` gọi `rainbow_prompt(...)` của từng layer. 
   - `RainbowPromptModule` stack prompt lịch sử và gọi `RainbowEvolution` để thu được prompt tiến hóa.
   - Gating trả về hệ số `gate_value` cho layer. Prompt được định dạng dạng prefix (key/value) và nhân với hệ số gate trước khi đưa vào attention block.
4. **Thu thập auxiliary loss**: `rainbow_aux` chứa logarit xác suất sử dụng từ gating để phục vụ sparse regularization.

### 4.4 Tính toán loss và tối ưu

- **Cross-entropy loss**: tính trên các lớp thuộc tác vụ hiện tại (đã offset nhãn).
- **Sparsity loss**: tổng hợp từ `rainbow_aux`, nhân với `lambda_sparse`.
- **Match loss**: nếu `lambda_match > 0`, `matcher.match_loss` áp ràng buộc giữa `pre_logits` và embedding tác vụ.
- Tổng loss được lan truyền ngược, gradient bị clip nếu cấu hình yêu cầu, sau đó optimizer cập nhật tham số.

### 4.5 Kết thúc tác vụ (`finalize_task`)

- Sau khi hoàn thành tất cả epoch, Rainbow Prompt lưu prompt và gate cuối cùng của từng layer vào storage.
- Pipeline tiến hành đánh giá (xem mục 7) và ghi checkpoint (nếu chỉ định).

## 5. Chi tiết tiến hóa prompt

### 5.1 Chế độ mặc định (`use_paper_evolution: false`)

- **Tính trọng số tác vụ**: nếu bật task conditioning, dự đoán `task_weights` bằng việc chiếu prompt lịch sử và embedding tác vụ, trộn với attention mức tác vụ.
- **Gộp đơn bước**: sử dụng Einstein summation để gộp keys, values và prompt nền thành các đại lượng “trọng số trung bình”.
- **Attention đặc trưng**: nếu bật `enable_feature_level`, thực hiện attention giữa truy vấn (prompt mới) và key gộp, thu về projection đã tiến hóa.
- **Căn chỉnh**: áp dụng `output_proj`, layer norm và mạng alignment để tạo prompt cuối cùng, đồng thời nhân bản kết quả cho mục đích logging.

Chế độ này tiết kiệm thời gian vì chỉ thực hiện một lần gộp, thích hợp cho thiết lập yêu cầu tốc độ.

### 5.2 Chế độ theo bài báo (`use_paper_evolution: true`)

- **Trọng số tác vụ**: giống chế độ mặc định.
- **Tiến hóa theo từng prompt**:
  - Nhân trọng số với từng key/value thay vì gộp ngay.
  - Lặp qua từng prompt lịch sử, thực hiện attention đặc trưng riêng, sau đó căn chỉnh.
- **Tổng hợp cuối**: lấy trung bình các prompt đã căn chỉnh để tạo `rainbow_prompt`.

Chế độ này giữ nhiều thông tin vi mô của từng prompt, nhưng chi phí tính toán lớn hơn.

## 6. Gating và regularization

`ProbabilisticGate` chứa logits kích thước `[num_layers, 2]` cho mỗi layer (trạng thái “tắt”/“bật”). Trong huấn luyện:

- Nhiệt độ $\tau$ được nội suy tuyến tính từ `tau_start` đến `tau_end` theo tiến độ epoch.
- Sử dụng Gumbel-Softmax để lấy mẫu phân phối mềm; sau ngưỡng `harden_epoch_ratio`, vector one-hot được sử dụng (straight-through).
- Sparsity loss là logarit của xác suất chọn trạng thái “bật”; càng âm thì càng khuyến khích tắt prompt.

Ở pha suy luận, gate chọn trạng thái có xác suất cao nhất (deterministic).

## 7. Quy trình suy luận và đánh giá

- **Trong huấn luyện**: sau mỗi tác vụ, hệ thống đánh giá tuần tự trên tất cả tác vụ đã học bằng cách nạp prompt từ storage và đặt Rainbow Prompt ở chế độ inference (`set_training(False)`).
- **Trong suy luận thuần**: khi gọi `evaluate_rainbow`, pipeline lấy prompt từ file `.pt` tương ứng, áp dụng gate đã lưu, sau đó chỉ forward trên lớp-lớp của tác vụ hiện hành.
- **Chỉ số báo cáo**: Accuracy top-1/top-5 và loss cho từng tác vụ; các số liệu này cũng dùng để tính forgetting/backward transfer.

## 8. So sánh hai chế độ tiến hóa

| Thuộc tính | `use_paper_evolution: false` | `use_paper_evolution: true` |
|------------|-------------------------------|-----------------------------|
| Chi phí tính toán | Thấp (một lần gộp) | Cao (lặp trên từng prompt lịch sử) |
| Bảo toàn thông tin lịch sử | Gián tiếp thông qua gộp | Trực tiếp, giữ cấu trúc từng prompt |
| Độ linh hoạt | Phù hợp với số lượng prompt lớn | Phù hợp khi cần độ chính xác cao |
| Hành vi gating | Giống nhau (gating nằm ngoài tiến hóa) | Giống nhau |

Khuyến nghị: bắt đầu với chế độ mặc định để đạt cân bằng hiệu năng – chi phí; chuyển sang chế độ bài báo khi dữ liệu đòi hỏi tiến hóa tinh vi hoặc để so sánh với kết quả công bố.

## 9. Kết luận

Rainbow Prompt cung cấp cách tiếp cận có hệ thống để khai thác prompt trong học liên tục:

- Khởi tạo linh hoạt theo tác vụ, bảo toàn lịch sử thông qua tiến hóa có điều kiện.
- Kết hợp gating xác suất nhằm tự động điều chỉnh lượng prompt cần thiết theo thời gian.
- Storage giúp suy luận và đánh giá giữ nguyên đặc tính đã học của từng tác vụ.

Hai chế độ tiến hóa cho phép cân đối giữa chi phí tính toán và độ trung thực với mô tả gốc, giúp phương pháp thích nghi với nhiều yêu cầu triển khai khác nhau.



