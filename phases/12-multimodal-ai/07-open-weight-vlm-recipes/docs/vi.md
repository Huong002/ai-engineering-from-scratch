# Công thức cho VLM open-weight (trọng số mở): điều gì thực sự quan trọng

> Tài liệu về VLM open-weight giai đoạn 2024-2026 là cả một khu rừng bảng ablation (thí nghiệm cắt bỏ từng biến). MM1 của Apple đã thử 13 tổ hợp encoder ảnh, connector và cách pha dữ liệu. Molmo của Allen AI chứng minh caption chi tiết do người viết đánh bại distillation từ GPT-4V. Cambrian-1 chạy so sánh hơn 20 encoder. Idefics2 hệ thống hoá không gian thiết kế năm trục. Prismatic VLMs so sánh 27 recipe (công thức huấn luyện) trên một benchmark có kiểm soát. Giữa tất cả sự nhiễu loạn đó, một tập kết quả nhỏ vẫn đúng xuyên suốt các bài báo: encoder ảnh quan trọng hơn kiến trúc connector, hỗn hợp dữ liệu (data mixture) quan trọng hơn cả hai, và caption chi tiết do người viết đánh bại dữ liệu tổng hợp đã chưng cất. Bài học này đọc hộ bạn những bảng đó.

**Type:** Learn + lab
**Languages:** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites:** Giai đoạn 12, Bài 05 (đường cơ sở LLaVA)
**Time:** ~180 phút

## Mục tiêu học tập

- Nêu tên không gian thiết kế năm trục của VLM: encoder ảnh, connector, LLM, cách pha dữ liệu, lịch độ phân giải.
- Đọc bảng ablation của MM1 / Idefics2 / Cambrian-1 và dự đoán nút vặn nào làm dịch chuyển một benchmark cho trước.
- Chọn một recipe (encoder, connector, dữ liệu, độ phân giải) cho VLM mới khi có ngân sách compute và hỗn hợp tác vụ.
- Giải thích vì sao caption chi tiết do người viết đánh bại distillation từ GPT-4V ở cùng số token.

## Vấn đề

Có hàng trăm VLM open-weight. Phần lớn khoảng cách giữa "tốt" và "state-of-the-art" không nằm ở kiến trúc. Nó nằm ở dữ liệu, lịch độ phân giải và lựa chọn encoder. Biết nên vặn nút nào trước khi mô hình của bạn chạy kém sẽ giúp bạn tránh một sai lầm tốn 5 triệu giờ GPU.

Làn sóng 2023 (LLaVA-1.5, InstructBLIP, MiniGPT-4) chạy trên pretraining cặp caption + LLaVA-Instruct-150k. Đường cơ sở tốt. Chạm trần quanh MMMU 35%.

Làn sóng 2024 (MM1, Idefics2, Molmo, Cambrian-1, Prismatic VLMs) chạy ablation toàn diện. Kết quả vừa bất ngờ vừa thực dụng.

## Khái niệm

### Không gian thiết kế năm trục

Idefics2 (Laurençon et al., 2024) đặt tên cho các trục:

1. Encoder ảnh. CLIP ViT-L/14, SigLIP SO400m/14, DINOv2 ViT-g/14, InternViT-6B. Các encoder khác nhau ở kích thước patch, độ phân giải và mục tiêu pretraining.
2. Connector. MLP (2-4 lớp), Q-Former (32 query + cross-attn), Perceiver Resampler (64 query), C-Abstractor (tích chập + pooling song tuyến).
3. Mô hình ngôn ngữ. Llama-3 8B / 70B, Mistral 7B, Phi-3, Gemma-2, Qwen2.5. Kích thước LLM là chi phí tham số chi phối.
4. Dữ liệu huấn luyện. Cặp caption (CC3M, LAION), dữ liệu xen kẽ (OBELICS, MMC4), dữ liệu chỉ dẫn (LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron).
5. Lịch độ phân giải. Cố định 224/336/448, AnyRes, native động. Tăng dần trong lúc huấn luyện hoặc giữ nguyên.

Mọi VLM thương mại đều phải đưa ra lựa chọn trên từng trục. Phần lớn phương sai điểm MMMU được giải thích bởi trục 1, 4 và 5 — không phải bởi connector bạn chọn.

### Trục 1: encoder > connector

MM1 Mục 3.2 chỉ ra: đổi từ CLIP ViT-L/14 sang SigLIP SO400m/14 thêm hơn 3 điểm MMMU. Đổi connector từ MLP sang Perceiver Resampler thêm chưa tới 1 điểm. Idefics2 lặp lại: SigLIP > CLIP, Q-Former ≈ MLP ≈ Perceiver ở cùng số token.

"Cambrian Vision Encoders Match-Up" của Cambrian-1 (Tong et al., 2024) chạy hơn 20 encoder trên một benchmark thiên về thị giác (CV-Bench). Đầu bảng là sự pha trộn DINOv2 và SigLIP; CLIP ở giữa; ImageBind và ViT-MAE thấp hơn. Khoảng cách từ CLIP ViT-L tới DINOv2 ViT-g/14 là ~5-7 điểm trên CV-Bench.

Encoder mặc định năm 2026 cho VLM mở là SigLIP 2 SO400m/14 cho đặc trưng ngữ nghĩa + đặc trưng dày, đôi khi được ghép nối với đặc trưng DINOv2 ViT-g/14 (cơ chế "Spatial Vision Aggregator" của Cambrian làm việc này).

### Trục 2: thiết kế connector gần như không tạo khác biệt

MM1, Idefics2, Prismatic và MM-Interleaved đều đi tới cùng kết luận: với số token thị giác cố định, kiến trúc connector gần như không quan trọng. Một MLP 2 lớp trên các patch đã mean-pool đạt trong khoảng 1 điểm so với Q-Former 32 query với cùng ngân sách token.

Thứ thực sự quan trọng là số token. Nhiều token thị giác hơn = nhiều compute cho LLM hơn = hiệu năng tốt hơn tới một điểm rồi lợi ích giảm dần. 64 token mỗi ảnh là quá ít cho OCR. 576-1024 token là vùng ngọt cho hầu hết VLM mở. 2048+ chỉ giúp cho tài liệu và biểu đồ.

Q-Former so với MLP là câu hỏi về chi phí, không phải chất lượng: Q-Former chặn số token ở 32-64 bất kể độ phân giải ảnh; MLP phát ra toàn bộ token patch. Với đầu vào độ phân giải cao, Q-Former tiết kiệm context của LLM; với độ phân giải thấp, khác biệt chỉ là nhiễu.

### Trục 3: kích thước LLM đặt ra trần

Tăng gấp đôi LLM từ 7B lên 13B một cách ổn định thêm 2-4 điểm MMMU trong mọi bài báo VLM. Ở 70B bạn bão hoà hầu hết benchmark. Trần suy luận đa phương thức của VLM chính là trần suy luận văn bản của LLM — encoder thị giác chỉ có thể cấp dữ liệu cho nó, không suy luận thay nó.

Đây là lý do Qwen2.5-VL-72B và Claude Opus 4.7 đè bẹp MMMU-Pro và ScreenSpot-Pro: bộ não ngôn ngữ rất lớn. Một VLM 7B không thể thay thế VLM 70B bằng thiết kế connector khéo léo.

### Trục 4: dữ liệu — caption chi tiết do người viết đánh bại distillation

Molmo + PixMo (Deitke et al., 2024) là kết quả năm 2024 mà ai cũng nên đọc. Allen AI cho người chú giải mô tả ảnh qua những lượt nói dày 1-3 phút rồi chuyển thành văn bản, thu được 712K ảnh có caption dày. Không có distillation từ GPT-4V ở bất kỳ đâu trong dữ liệu huấn luyện.

Molmo-72B đánh bại Llama-3.2-90B-Vision trên 11/11 benchmark. Chênh lệch không phải kiến trúc — đó là chất lượng caption. Caption chi tiết do người viết chứa thông tin nhiều gấp 5-10 lần mỗi ảnh so với caption web ngắn và vẫn bám đúng thực tế, trong khi distillation từ GPT-4V sinh ra ảo giác.

ShareGPT4V (Chen et al., 2023) và Cauldron (Idefics2) đi theo cùng một lối với caption pha trộn người viết + GPT-4V. Xu hướng rõ ràng: ở biên giới năm 2026, độ dày caption > số lượng caption > sự tiện lợi của distillation.

### Trục 5: độ phân giải và lịch của nó

Ablation của Idefics2: 384 -> 448 thêm 1-2 điểm. 448 -> 980 với chia ảnh (AnyRes) thêm 3-5 điểm nữa trên các benchmark OCR. Huấn luyện ở độ phân giải phẳng bão hoà ở mức chính xác trung bình; tăng dần độ phân giải (bắt đầu 224, kết thúc 448 hoặc native) huấn luyện nhanh hơn và kết thúc cao hơn.

Cambrian-1 chạy thử đánh đổi độ phân giải so với token: với compute cố định, bạn có thể có nhiều token hơn ở độ phân giải thấp hơn hoặc ít token hơn ở độ phân giải cao hơn. Độ phân giải cao hơn thắng cho OCR; độ phân giải thấp hơn-nhiều token hơn thắng cho hiểu cảnh tổng quát.

Công thức sản xuất năm 2026: huấn luyện chặng huấn luyện (training stage) 1 ở 384 cố định, chặng 2 với độ phân giải động tới 1280 cho các tác vụ nặng OCR.

### So sánh có kiểm soát của Prismatic

Prismatic VLMs (Karamcheti et al., 2024) là bài báo kiểm soát mọi trục. Cùng LLM 13B, cùng dữ liệu chỉ dẫn, cùng đánh giá — mỗi lần chỉ một trục thay đổi. Kết quả:

- Số token thị giác mỗi ảnh giải thích ~60% phương sai.
- Lựa chọn encoder giải thích ~20%.
- Kiến trúc connector giải thích ~5%.
- Mọi thứ còn lại (hỗn hợp dữ liệu, scheduler, LR) chiếm ~15% còn lại.

Đây là một phân rã thô, nhưng nó là câu trả lời gọn nhất cho câu hỏi "nên ablation cái gì trước" trong tài liệu.

### Bộ chọn cho năm 2026

Dựa trên bằng chứng thu được, recipe mặc định cho VLM mở của một dự án mới năm 2026:

- Encoder: SigLIP 2 SO400m/14 ở độ phân giải native với NaFlex, ghép nối với DINOv2 ViT-g/14 cho đặc trưng dày nếu bạn cần phân đoạn/grounding.
- Connector: MLP 2 lớp trên token patch. Bỏ qua Q-Former trừ khi bạn bị giới hạn token.
- LLM: Qwen2.5 / Llama-3.1 / Gemma 2, 7B cho chi phí, 70B cho chất lượng, chọn theo độ trễ mục tiêu.
- Dữ liệu: PixMo + ShareGPT4V + Cauldron, bổ sung dữ liệu chỉ dẫn theo tác vụ.
- Độ phân giải: động (tối thiểu 256, tối đa 1280 pixel cho cạnh dài).
- Lịch: chặng 1 căn chỉnh (chỉ projector), chặng 2 fine-tune toàn bộ, chặng 3 fine-tune theo tác vụ.

Mỗi mặc định trong số đó đều truy về một ablation đã được đo trong các bài báo trích ở cuối bài học này.

```figure
l5-vlm-recipe-knobs
```

## Áp dụng

`code/main.py` là bộ phân tích bảng ablation và bộ chọn recipe. Nó mã hoá bảng ablation của MM1 và Idefics2 (đã rút gọn) và cho bạn truy vấn:

- "Với ngân sách X và tác vụ Y, recipe nào thắng?"
- "Nếu tôi đổi SigLIP thành CLIP trên Llama 7B, chênh lệch MMMU dự kiến là bao nhiêu?"
- "Tôi nên ablation trục nào trước để có câu trả lời với độ tin cậy 80%?"

Đầu ra là một danh sách recipe đã xếp hạng với chênh lệch benchmark dự kiến và một khuyến nghị "ablation trước".

## Đóng gói

Bài học này tạo ra `outputs/skill-vlm-recipe-picker.md`. Cho trước một hỗn hợp tác vụ mục tiêu, một ngân sách compute và một mục tiêu độ trễ, nó xuất ra một recipe đầy đủ (encoder, connector, LLM, hỗn hợp dữ liệu, lịch độ phân giải) kèm trích dẫn tới ablation biện minh cho từng lựa chọn. Nó ngăn kỹ sư phải phát minh lại bảng ablation Idefics2 mỗi khi một dự án VLM mới khởi động.

## Bài tập

1. Đọc MM1 Mục 3.2. Với LLM 2B cố định và ngân sách 50M ảnh, encoder nào thắng? Câu trả lời có đảo chiều ở LLM 13B không? Vì sao?

2. Cambrian-1 thấy rằng ghép DINOv2 + SigLIP tốt hơn từng cái riêng lẻ trên các benchmark thiên thị giác nhưng không thêm tín hiệu trên MMMU. Dự đoán benchmark nào được lợi và cái nào giữ nguyên.

3. Mục tiêu của bạn là một agent UI di động trên LLM 2B. Chọn encoder, connector, độ phân giải và hỗn hợp dữ liệu. Biện minh từng lựa chọn bằng một bảng ablation cụ thể.

4. Molmo phát hành mô hình 4B và 72B. Bản 4B cạnh tranh với các VLM đóng 7B; bản 72B đánh bại Llama-3.2-90B-Vision trên 11/11 benchmark. Điều đó nói gì về giả thuyết bão hoà theo kích thước LLM?

5. Thiết kế một bảng ablation để tách chất lượng hỗn hợp dữ liệu khỏi chất lượng encoder trên một VLM 7B. Tối thiểu bao nhiêu lần huấn luyện? Đề xuất bốn thiết lập trục.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Ablation | "Vặn một nút" | Huấn luyện nhiều lần chạy chỉ khác nhau đúng một trục của không gian thiết kế, giữ mọi thứ khác cố định |
| Connector | "Cầu nối" / "projector" | Mô-đun huấn luyện được ánh xạ đầu ra của encoder thị giác vào không gian token của LLM (MLP, Q-Former, Perceiver) |
| Detailed human caption | "Caption dày" | Mô tả nhiều câu do người viết (thường 80-300 token) giàu thông tin hơn alt text trên web |
| Distillation | "Caption GPT-4V" | Dữ liệu huấn luyện được sinh bởi một VLM độc quyền mạnh hơn; tiện nhưng dễ mang theo ảo giác có sẵn |
| AnyRes / dynamic res | "Đường độ phân giải cao" | Chiến lược đưa ảnh lớn hơn độ phân giải native của encoder vào qua chia ô hoặc M-RoPE |
| Resolution ramp | "Chương trình học" | Lịch huấn luyện bắt đầu ở độ phân giải thấp rồi tăng dần, giúp học căn chỉnh nhanh hơn |
| Vision-centric bench | "CV-Bench / BLINK" | Đánh giá nhấn vào tri giác thị giác chi tiết thay vì suy luận nặng về ngôn ngữ |
| PixMo | "Dữ liệu của Molmo" | Bộ dữ liệu 712K ảnh có caption dày của Allen AI; lời nói của người được chuyển thành caption dày |

## Đọc thêm

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)

---

[← Bài trước: Thị giác any-resolution (độ phân giải tuỳ ý): Patch-n'-Pack và NaFlex](../../06-any-resolution-patch-n-pack/docs/vi.md) · [Bài sau: LLaVA-OneVision: Một ảnh, nhiều ảnh và video trong cùng một mô hình →](../../08-llava-onevision-single-multi-video/docs/vi.md)
