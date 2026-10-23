# Vision Transformer (ViT) và primitive patch-token

> Trước mọi thứ đa phương thức, một ảnh phải trở thành chuỗi token (đơn vị biểu diễn) mà transformer có thể ăn được. Bài báo ViT năm 2020 trả lời bằng patch (mảnh ảnh) 16x16 pixel, một phép chiếu tuyến tính, và một position embedding (embedding vị trí). Năm năm sau, mọi mô hình tiên phong của 2026 (Claude Opus 4.7 ở 2576px native, Gemini 3.1 Pro, Qwen3.5-Omni) vẫn bắt đầu theo cách này — encoder đổi từ ViT sang DINOv2 rồi SigLIP 2, register token được thêm vào, sơ đồ vị trí trở thành 2D-RoPE, nhưng primitive vẫn giữ nguyên. Bài này đọc toàn bộ pipeline patch-token từ đầu đến cuối và tự dựng nó bằng Python stdlib để phần còn lại của Giai đoạn 12 có một hình mẫu tinh thần cụ thể cho "visual tokens."

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Giai đoạn 7 (Transformer), Giai đoạn 4 (Thị giác máy tính)
**Time:** ~120 phút

## Mục tiêu học tập

- Chuyển một ảnh HxWx3 thành chuỗi patch token với positional encoding chính xác.
- Tính độ dài chuỗi, số tham số, và FLOPs cho một ViT với (patch size, resolution, hidden dim, depth) cho trước.
- Kể tên ba nâng cấp đưa ViT từ nghiên cứu năm 2020 tới production năm 2026: pretraining tự giám sát (DINO / MAE), register token, và đóng gói ở độ phân giải gốc.
- Chọn giữa CLS pooling, mean pooling, và register token cho một tác vụ hạ nguồn.

## Vấn đề

Transformer vận hành trên chuỗi vector. Văn bản vốn đã là một chuỗi (byte hoặc token). Một ảnh là lưới pixel 2D với ba kênh màu — không phải chuỗi. Nếu bạn làm phẳng từng pixel, một ảnh RGB 224x224 trở thành 150,528 token, và self-attention ở độ dài đó là bất khả thi (bậc hai theo độ dài chuỗi).

Các cách làm trước 2020 gắn một CNN feature extractor lên phía trước: ResNet tạo ra feature map 7x7 gồm các vector 2048 chiều, đưa 49 token đó vào transformer. Cách này chạy được nhưng thừa hưởng thiên kiến của CNN (translation equivariance, trường tiếp nhận cục bộ) và đánh mất khả năng thèm quy mô của transformer.

Dosovitskiy và cộng sự (2020) đặt câu hỏi thẳng: nếu bỏ luôn CNN thì sao? Chia ảnh thành các patch kích thước cố định (ví dụ 16x16 pixel), chiếu tuyến tính từng patch thành một vector, cộng thêm một positional embedding, rồi đưa chuỗi đó vào một transformer vanilla. Thời điểm đó đây là điều dị giáo — thị giác mà không có tích chập. Với đủ dữ liệu (JFT-300M, rồi LAION) nó đánh bại ResNet trên ImageNet và tiếp tục cải thiện.

Đến 2026, primitive ViT là nền tảng không còn gì phải bàn. Vision tower của mọi VLM open-weights đều là một hậu duệ nào đó (DINOv2, SigLIP 2, CLIP, EVA, InternViT). Câu hỏi không còn là "có nên dùng patch không?" mà là "patch size nào, lịch resolution nào, mục tiêu pretraining nào, positional encoding nào."

## Khái niệm

### Patch là token

Cho một ảnh `x` có shape `(H, W, 3)` và một patch size `P`, bạn cắt ảnh thành lưới `(H/P) x (W/P)` các patch không chồng lấn. Mỗi patch là một khối pixel `P x P x 3`. Làm phẳng mỗi khối thành một vector `3 P^2`. Áp một phép chiếu tuyến tính dùng chung `W_E` có shape `(3 P^2, D)` để ánh xạ từng patch vào chiều ẩn `D` của mô hình.

Với cấu hình chuẩn ViT-B/16:
- resolution (độ phân giải) 224, patch size 16 → lưới 14x14 → 196 patch token.
- Mỗi patch là `16 x 16 x 3 = 768` giá trị pixel, được chiếu thành `D = 768`.
- Thêm một token `[CLS]` học được → độ dài chuỗi 197.

Phép chiếu patch — tức patch embedding (embedding của mảnh ảnh) — về mặt toán học giống hệt một phép tích chập 2D với kernel size `P`, stride `P`, và `D` kênh đầu ra. Đó chính là cách code production triển khai nó — `nn.Conv2d(3, D, kernel_size=P, stride=P)`. Cách gọi "phép chiếu tuyến tính" là khái niệm; cách gọi "kernel" là hiệu quả.

### Positional embedding

Patch không có thứ tự cố hữu — transformer thấy chúng như một túi. ViT thời đầu thêm một positional embedding 1D học được (một vector 768 chiều cho mỗi vị trí, 197 vector). Chạy được, nhưng buộc mô hình vào resolution huấn luyện: khi suy luận bạn phải nội suy bảng vị trí nếu đổi lưới.

Các vision backbone hiện đại dùng 2D-RoPE (M-RoPE của Qwen2-VL, mặc định của SigLIP 2) hoặc vị trí 2D phân rã. 2D-RoPE xoay các vector query và key dựa trên chỉ số (hàng, cột) của patch, nhờ đó mô hình suy ra vị trí 2D tương đối từ góc xoay. Không cần bảng vị trí. Mô hình xử lý được lưới kích thước tuỳ ý khi suy luận.

### CLS token, pooled output, và register token

Đâu là biểu diễn mức ảnh? Ba lựa chọn cùng tồn tại:

1. token `[CLS]`. Gắn thêm một vector học được vào đầu chuỗi patch. Sau tất cả khối transformer, trạng thái ẩn của token CLS là biểu diễn ảnh. Thừa hưởng từ BERT. Được ViT gốc và CLIP dùng.
2. Mean pool. Lấy trung bình trạng thái ẩn đầu ra của các patch token. Được SigLIP, DINOv2, và hầu hết VLM hiện đại dùng.
3. Register token. Darcet và cộng sự (2023) quan sát thấy ViT huấn luyện mà không có token sink rõ ràng sẽ sinh ra các patch "artifact" chuẩn lớn chiếm dụng self-attention. Thêm 4–16 register token học được sẽ hấp thụ tải này và cải thiện chất lượng dự đoán dày (segmentation, depth). DINOv2 và SigLIP 2 đều có register.

Lựa chọn này quan trọng với tác vụ hạ nguồn. CLS ổn cho phân loại. Với VLM đưa patch token vào LLM, bạn bỏ hẳn pooling — mỗi patch trở thành một input token của LLM. Register bị loại trước khi bàn giao (chúng là giàn giáo, không phải nội dung).

### Pretraining: có giám sát, tương phản, che khuất, tự chưng cất

ViT năm 2020 được pretraining bằng phân loại có giám sát trên JFT-300M. Nhanh chóng bị thay thế bởi:

- CLIP (2021): tương phản ảnh-văn bản trên 400M cặp. Bài 12.02.
- MAE (2021, He và cộng sự): che 75% patch, tái tạo pixel. Tự giám sát, chạy trên ảnh thuần.
- DINO (2021) / DINOv2 (2023): tự chưng cất với student-teacher, không nhãn, không caption. DINOv2 ViT-g/14 năm 2023 là backbone thuần thị giác mạnh nhất và là mặc định cho các trường hợp dùng "dense features".
- SigLIP / SigLIP 2 (2023, 2025): CLIP với loss sigmoid và NaFlex cho tỉ lệ khung gốc. Vision tower thống trị trong VLM mở 2026 (Qwen, Idefics2, LLaVA-OneVision).

Lựa chọn pretraining quyết định backbone giỏi việc gì: CLIP/SigLIP cho khớp ngữ nghĩa với văn bản, DINOv2 cho đặc trưng thị giác dày, MAE làm điểm khởi đầu để fine-tuning hạ nguồn.

### Định luật scaling

Scaling ViT (Zhai và cộng sự 2022) xác lập rằng chất lượng ViT tuân theo những định luật dự đoán được về kích thước mô hình, kích thước dữ liệu, và compute. Với compute cố định:
- Mô hình lớn hơn + nhiều dữ liệu hơn → chất lượng tốt hơn.
- Patch size là đòn bẩy giữa độ dài chuỗi và độ trung thực. Patch 14 (thường dùng cho DINOv2/SigLIP SO400m) cho nhiều token hơn trên mỗi ảnh so với patch 16; tốt hơn cho OCR và tác vụ dày, tệ hơn về tốc độ.
- Resolution là đòn bẩy lớn còn lại. Đi từ 224 lên 384 rồi 512 gần như luôn giúp, với chi phí FLOPs tăng theo bậc hai.

ViT-g/14 (1B tham số, patch 14, resolution 224 → 256 token) và SigLIP SO400m/14 (400M tham số, patch 14) là hai encoder chủ lực cho VLM mở 2026.

### Số tham số của một ViT

Toàn bộ phép tính nằm trong `code/main.py`. Với ViT-B/16 ở 224:

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

Hãy ước lượng sơ bộ mọi ViT theo cách này trước khi nạp checkpoint. Kích thước backbone đặt sàn VRAM của bạn trong mọi VLM hạ nguồn.

### Cấu hình production 2026

Encoder mà hầu hết VLM mở mang theo trong 2026 là SigLIP 2 SO400m/14 ở độ phân giải gốc (NaFlex). Nó có:
- 400M tham số.
- Patch size 14, resolution mặc định 384 → 729 patch token mỗi ảnh.
- Mean pool cho tác vụ mức ảnh; cả 729 patch chảy vào LLM cho VQA.
- 4 register token, bị loại trước khi bàn giao cho LLM.
- 2D-RoPE với scaling mức ảnh cho tỉ lệ khung gốc.

Mọi quyết định trong cấu hình đó đều truy về một bài báo bạn có thể đọc.

```figure
image-patch-tokens
```

## Áp dụng

`code/main.py` là một patch tokenizer và máy tính hình học. Nó nhận (ảnh H, W, patch P, hidden D, depth L) và báo cáo:

- Shape lưới và độ dài chuỗi sau khi patch.
- Chuỗi token cho một ảnh đồ chơi tổng hợp 8x8 pixel (đi qua đường làm phẳng + chiếu).
- Số tham số chia theo patch embed, position embed, khối transformer, và classification head (đầu phân loại).
- FLOPs mỗi lượt forward ở resolution đích.
- Một bảng so sánh giữa ViT-B/16 @ 224, ViT-L/14 @ 336, DINOv2 ViT-g/14 @ 224, SigLIP SO400m/14 @ 384.

Chạy nó. Đối chiếu số tham số với các con số đã công bố. Thử đổi patch size và resolution để cảm nhận chi phí số token.

## Đóng gói

Bài này tạo ra `outputs/skill-patch-geometry-reader.md`. Cho một cấu hình ViT (patch size, resolution, hidden dim, depth), nó đưa ra ước lượng số token, số tham số, và VRAM kèm lý do. Dùng skill này mỗi khi bạn chọn vision backbone cho một VLM — nó tránh được cú sốc "token nổ tung và context LLM của tôi đầy mất".

## Bài tập

1. Tính độ dài chuỗi patch-token cho Qwen2.5-VL ở đầu vào gốc 1280x720 với patch size 14. So với biểu diễn chỉ dùng CLS thì thế nào?

2. Một khung 1080p (1920x1080) ở patch 14 tạo ra bao nhiêu token? Ở 30 FPS trong video 5 phút, tổng cộng bao nhiêu visual token? Chi phí nào tiết kiệm nhất: pooling, lấy mẫu khung, hay token merging?

3. Cài mean pooling trên patch token bằng Python thuần. Kiểm chứng rằng mean-pool trên 196 token của đầu ra DINOv2 khớp với cái `forward` của mô hình trả về khi bạn yêu cầu một embedding đã pool.

4. Đọc Phần 3 của "Vision Transformers Need Registers" (arXiv:2309.16588). Mô tả trong hai câu register hấp thụ artifact gì và vì sao điều đó quan trọng cho dự đoán dày hạ nguồn.

5. Sửa `code/main.py` để hỗ trợ patch-n'-pack: cho một danh sách ảnh có resolution khác nhau, tạo ra một chuỗi đã đóng gói duy nhất và attention mask block-diagonal. Kiểm chứng với Bài 12.06 khi bạn tới đó.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Patch | "16x16 pixel square" | Một vùng không chồng lấn kích thước cố định của ảnh đầu vào; trở thành một token |
| Patch embedding | "Linear projection" | Một ma trận học được dùng chung (hoặc Conv2d với stride=P) ánh xạ pixel patch đã làm phẳng thành vector D chiều |
| CLS token | "Class token" | Vector học được gắn lên đầu chuỗi, trạng thái ẩn cuối của nó biểu diễn toàn bộ ảnh; tuỳ chọn vào 2026 |
| Register token | "Sink token" | Token học được thêm vào để hấp thụ các artifact attention chuẩn lớn mà ViT sinh ra trong pretraining |
| Position embedding | "Positional info" | Vector hoặc phép xoay theo từng vị trí làm chuỗi nhận biết được thứ tự; 2D-RoPE là mặc định hiện đại |
| Grid | "Patch grid" | Mảng 2D (H/P) x (W/P) các patch cho một resolution và patch size cho trước |
| NaFlex | "Native flexible resolution" | Tính năng của SigLIP 2: một mô hình phục vụ nhiều tỉ lệ khung và resolution mà không cần huấn luyện lại |
| Backbone | "Vision tower" | Encoder ảnh đã pretraining, đầu ra patch-token của nó được đưa vào LLM trong một VLM |
| Pooling | "Image-level summary" | Chiến lược biến patch token thành một vector: CLS, mean, attention pool, hoặc dựa trên register |
| Patch 14 vs 16 | "Finer vs coarser grid" | Patch 14 tạo nhiều token hơn mỗi ảnh, trung thực hơn cho OCR, chậm hơn; patch 16 là mặc định cổ điển |

## Đọc thêm

- [Dosovitskiy và cộng sự — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — ViT gốc.
- [He và cộng sự — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE, pretraining tự giám sát.
- [Oquab và cộng sự — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — tự chưng cất ở quy mô lớn, không nhãn.
- [Darcet và cộng sự — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register token và phân tích artifact.
- [Tschannen và cộng sự — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — vision tower mặc định của 2026.
- [Zhai và cộng sự — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — định luật scaling thực nghiệm.

---

[Bài sau: CLIP và tiền huấn luyện tương phản thị giác-ngôn ngữ →](../../02-clip-contrastive-pretraining/docs/vi.md)
