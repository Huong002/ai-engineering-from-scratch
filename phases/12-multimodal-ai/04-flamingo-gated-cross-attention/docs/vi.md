# Flamingo và gated cross-attention (cross-attention có cổng) cho VLM few-shot

> Flamingo của DeepMind (2022) làm được hai điều trước tất cả mọi người. Nó cho thấy một mô hình duy nhất có thể xử lý chuỗi hình ảnh, video và văn bản xen kẽ (interleaved) tuỳ ý. Và nó cho thấy VLM có thể học trong ngữ cảnh — đưa một prompt few-shot (học với vài ví dụ) gồm ba cặp (hình, chú thích) mẫu và mô hình sẽ chú thích một hình mới mà không cần bước gradient nào. Cơ chế: các lớp gated cross-attention, chèn giữa các lớp sẵn có của LLM đang bị đóng băng, với một tanh gate (cổng tanh) học được khởi tạo bằng không để năng lực văn bản của LLM được giữ nguyên lúc khởi tạo. Bài này đi qua kiến trúc perceiver resampler (bộ tái lấy mẫu kiểu Perceiver) và gated cross-attention của Flamingo — tổ tiên của đầu vào xen kẽ ở Gemini và visual token (token thị giác) của Idefics2.

**Type:** Learn
**Languages:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites:** Giai đoạn 12, Bài 03 (BLIP-2 Q-Former)
**Time:** ~120 phút

## Mục tiêu học tập

- Giải thích cách gated cross-attention giữ nguyên năng lực văn bản của một LLM đóng băng lúc khởi tạo nhờ tanh(gate) = 0.
- Đi qua một perceiver resampler: N patch ảnh → K truy vấn "latent" cố định thông qua cross-attention.
- Mô tả cách Flamingo xử lý chuỗi hình-văn bản xen kẽ với causal masking tôn trọng vị trí hình ảnh.
- Tái tạo cấu trúc prompt đa phương thức few-shot (3 ví dụ hình-chú thích rồi một hình truy vấn).

## Vấn đề

BLIP-2 đưa 32 visual token vào lớp đầu vào của một LLM đóng băng. Cách này hiệu quả với một hình cho mỗi prompt. Nhưng nếu bạn muốn đưa *nhiều* hình xen kẽ với văn bản, như trong "đây là hình A, chú thích nó; đây là hình B, chú thích nó; giờ đây là hình C, chú thích nó" thì sao? Self-attention của LLM sẽ phải xử lý token hình và token văn bản trong cùng một luồng, và câu hỏi vị trí nào được chú ý tới hình nào trở nên rối rắm.

Câu trả lời của Flamingo: không thay đổi luồng đầu vào của LLM chút nào. Chèn thêm các lớp cross-attention giữa các khối LLM sẵn có. Token văn bản vẫn chảy qua causal self-attention của LLM như thường lệ. Cứ vài khối LLM một lần, token văn bản cũng cross-attend tới đặc trưng hình ảnh qua một lớp có cổng (gate) mới. Cổng (khởi tạo bằng không) nghĩa là ở bước không, các lớp mới là no-op — mô hình hành xử đúng như LLM đã huấn luyện trước. Khi huấn luyện tiến triển, cổng mở ra và thông tin thị giác bắt đầu chảy vào.

Câu hỏi thứ hai Flamingo trả lời: làm sao xử lý số lượng hình thay đổi (0, 1, hoặc nhiều) mỗi prompt? Một perceiver resampler — mô-đun cross-attention nhỏ nhận bao nhiêu patch bạn có và tạo ra một số cố định token latent thị giác. Lớp cross-attention của LLM thấy cùng một hình dạng bất kể prompt có bao nhiêu hình.

## Khái niệm

### LLM đóng băng

Flamingo bắt đầu với một LLM Chinchilla 70B đóng băng. Toàn bộ 70B trọng số không đổi. Self-attention văn bản và FFN sẵn có hoạt động bình thường.

### Perceiver resampler

Với mỗi hình trong prompt, ViT tạo ra N patch token. Perceiver resampler có K latent học được cố định (Flamingo dùng K=64). Mỗi khối resampler gồm hai bước con:

1. Cross-attention: K latent chú ý trên N patch token (Q từ latent, K/V từ patch).
2. Self-attention + FFN trong nội bộ các latent.

Sau 6 khối resampler, đầu ra là K=64 visual token chiều 1024, bất kể ViT tạo ra bao nhiêu patch. Một hình 224x224 (196 patch) và một hình 480x480 (900 patch) đều ra 64 token resampler.

Với video, resampler được áp dụng theo thời gian: patch của mỗi khung tạo ra 64 latent, và một temporal positional encoding cho phép mô hình phân biệt t=0 với t=N. Toàn bộ video trở thành T * 64 visual token.

### Gated cross-attention

Cứ mỗi M lớp của LLM đóng băng (Flamingo dùng M=4), chèn một khối gated cross-attention mới:

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` là một scalar học được khởi tạo bằng không.
- `tanh(0) = 0`, nên lúc khởi tạo nhánh có cổng đóng góp bằng không.
- Khi `alpha` rời xa số không, phần đóng góp cross-attention tăng trơn tru.
- Kết nối residual nghĩa là ngay cả khi cổng mở hoàn toàn cũng không ghi đè biểu diễn văn bản của LLM; nó chỉ thêm thông tin thị giác lên trên.

Đây là lựa chọn thiết kế quan trọng nhất trong Flamingo: điều kiện hoá thị giác mang tính cộng thêm, có cổng, và bằng không lúc khởi tạo. Một Flamingo ở bước 0 là một Chinchilla 70B hoàn hảo trên đầu vào chỉ có văn bản.

### Cross-attention có mask cho đầu vào xen kẽ

Trong một prompt như "<image A> caption A <image B> caption B <image C> ?", mỗi token văn bản chỉ nên thấy những hình đến trước nó trong chuỗi. Mask của cross-attention bắt buộc: token văn bản ở vị trí `t` chỉ chú ý tới các token resampler của hình có chỉ số `i < i_t`, trong đó `i_t` là hình gần nhất trước vị trí `t`. "Chỉ thấy hình đứng ngay trước" hay "thấy tất cả hình đứng trước" đều là lựa chọn hợp lệ; Flamingo chọn cách đầu.

### Học few-shot trong ngữ cảnh

Một prompt của Flamingo trông như:

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

Mô hình thấy mẫu hoàn thành và xuất ra "bird" (hoặc bất cứ gì hình 3 hiển thị). Không có bước gradient nào. Năng lực học trong ngữ cảnh của LLM đóng băng được truyền qua gated cross-attention — đây là điểm nhấn của bài báo và là lý do nó quan trọng.

### Dữ liệu huấn luyện

Flamingo huấn luyện trên ba bộ dữ liệu:

1. MultiModal MassiveWeb (M3W): 43M trang web với hình và văn bản xen kẽ, tái dựng thứ tự đọc.
2. Image-Text Pairs (ALIGN + LTIP): 4.4B cặp.
3. Video-Text Pairs (VTP): 27M clip video ngắn.

OBELICS (2023) là bản tái tạo mở của kho văn bản web xen kẽ, thứ mà Idefics, Idefics2 và hầu hết mô hình mở "giống Flamingo" huấn luyện trên đó.

### OpenFlamingo và Otter

OpenFlamingo (2023) là bản tái tạo mở. Kiến trúc giống hệt (Perceiver resampler + gated cross-attention trên LLaMA hoặc MPT đóng băng). Checkpoint ở 3B, 4B, 9B. Chất lượng thua Flamingo do LLM nền nhỏ hơn và ít dữ liệu hơn.

Otter (2023) xây trên OpenFlamingo với instruction tuning trên MIMIC-IT (một bộ dữ liệu chỉ dẫn đa phương thức), cho thấy gated cross-attention cũng hiệu quả với việc tuân theo chỉ dẫn.

### Các hậu duệ

- Idefics / Idefics2 / Idefics3: dòng gated cross-attention của Hugging Face, ngày càng đơn giản hơn (Idefics2 bỏ resampler để dùng thẳng patch token với adaptive pooling).
- Chuyển dịch Flamingo-sang-Chameleon: đến 2024 nhiều nhóm chuyển sang early-fusion (Bài 12.11); gated cross-attention kiểu Flamingo vẫn còn trong môi trường production ở nơi cần đóng băng backbone.
- Đầu vào xen kẽ của Gemini: về mặt khái niệm thừa hưởng sự linh hoạt định dạng xen kẽ của Flamingo, dù cơ chế chính xác là độc quyền.

### So sánh với BLIP-2

| | BLIP-2 | Flamingo |
|---|---|---|
| Cầu nối thị giác | Q-Former một lần ở đầu vào | Gated cross-attention ở mỗi M lớp |
| Visual token | 32 mỗi hình | 64 mỗi hình mỗi lớp cross-attn |
| LLM đóng băng | Có | Có |
| Few-shot trong ngữ cảnh | Yếu | Mạnh — điểm nhấn của bài báo |
| Đầu vào xen kẽ | Không hỗ trợ gốc | Có, mục tiêu thiết kế |
| Dữ liệu huấn luyện | 130M cặp | 1.3B cặp + 43M trang xen kẽ |
| Số tham số | 188M được huấn luyện | ~10B được huấn luyện (lớp cross-attn) |
| Compute | Vài ngày trên 8 A100 | Vài tuần trên hàng nghìn TPUv4 |

Chọn BLIP-2 cho VQA một hình với ngân sách hạn chế. Chọn Flamingo/Idefics2 cho suy luận xen kẽ, few-shot, hoặc nhiều hình.

```figure
cross-attention-fusion
```

## Áp dụng

`code/main.py` minh hoạ:

1. Một Perceiver resampler trên 36 patch token giả với 8 latent học được (cross-attention thuần Python).
2. Một bước gated cross-attention với `alpha = 0` → đầu ra bằng đầu vào (LLM không đổi), rồi `alpha = 2.0` → đóng góp thị giác được trộn vào.
3. Một bộ dựng mask xen kẽ tạo mask attention 2D cho chuỗi "(image 1) (text 1) (image 2) (text 2)".

## Đóng gói

Bài này tạo ra `outputs/skill-gated-bridge-diagnostic.md`. Cho một config của VLM mở (resampler Có/Không, tần suất cross-attn, kiểu cổng), nó nhận diện các yếu tố thuộc dòng Flamingo và giải thích chiến lược đóng băng. Hữu ích để gỡ lỗi tại sao một lần fine-tune làm giảm hiệu năng văn bản (trả lời: cổng mở quá rộng quá nhanh).

## Bài tập

1. Tính số tham số thị giác của Flamingo-9B: 9B LLM + 1.4B lớp gated cross-attention + 64M resampler. Phần nào trong tổng tham số được huấn luyện?

2. Cài đặt residual có cổng `y = tanh(alpha) * cross + x` trong PyTorch. Chứng minh bằng thực nghiệm rằng với `alpha=0`, `y==x` đúng chính xác lúc khởi tạo.

3. Đọc Mục 3.2 của OpenFlamingo (arXiv:2308.01390) về cách họ xử lý nhiều hình trong một batch khi mỗi prompt có số hình khác nhau. Mô tả chiến lược padding.

4. Vì sao mask cross-attention của Flamingo cho một token văn bản chú ý tới *chỉ hình đứng trước gần nhất* thay vì tất cả hình đứng trước? Đọc Mục 2.4 của bài báo Flamingo và giải thích đánh đổi.

5. Few-shot trong ngữ cảnh: dựng một prompt với 4 ví dụ "hình → màu của vật chính" cho một biến thể Flamingo mới. Mô tả mẫu độ chính xác kỳ vọng khi bạn thay đổi số ví dụ từ 0 đến 8.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Perceiver resampler | "Cross-attention latent cố định" | Mô-đun tạo ra K token cố định từ số patch đầu vào thay đổi |
| Gated cross-attention | "Cầu nối có cổng tanh" | Lớp residual `y = tanh(alpha)*cross + x`, alpha học được, khởi tạo 0 |
| Interleaved input | "Chuỗi trộn lẫn" | Định dạng prompt với hình và văn bản trộn tự do theo thứ tự đọc |
| Frozen LLM | "Không gradient cho LLM" | Trọng số của LLM văn bản không cập nhật; chỉ resampler + lớp cross-attn được huấn luyện |
| Few-shot | "Ví dụ trong ngữ cảnh" | Đưa vài cặp (hình, đáp án) vào prompt; mô hình khái quát mà không cần finetuning |
| OBELICS | "Kho văn bản web xen kẽ" | Bộ dữ liệu mở gồm 141M trang web với hình và văn bản theo thứ tự đọc |
| Chinchilla | "Nền 70B đóng băng" | LLM văn bản đóng băng của Flamingo, từ bài báo Chinchilla của DeepMind |
| Gate schedule | "Cách alpha di chuyển" | Tốc độ mà cổng cross-attention mở ra trong quá trình huấn luyện |
| Cross-attn frequency | "Mỗi M lớp" | Tần suất chèn một khối gated cross-attention; Flamingo dùng M=4 |
| OpenFlamingo | "Bản tái tạo mở" | Checkpoint mở của MosaicML/LAION ở 3-9B; kiến trúc giống hệt Flamingo |

## Đọc thêm

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — bài báo gốc.
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — bản tái tạo mở.
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — kho văn bản web xen kẽ.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — kiến trúc Perceiver tổng quát.
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — hậu duệ Flamingo được tinh chỉnh chỉ dẫn.
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — bản đơn giản hoá hiện đại của cách tiếp cận Flamingo.

---

[← Bài trước: Từ CLIP đến BLIP-2](../../03-blip2-qformer-bridge/docs/vi.md) · [Bài sau: LLaVA và tinh chỉnh theo chỉ dẫn thị giác →](../../05-llava-visual-instruction-tuning/docs/vi.md)
