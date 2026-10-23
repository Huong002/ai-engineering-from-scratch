# Attention sai phân (V2)

> Attention softmax rải một lượng xác suất nhỏ lên mọi token không khớp. Với 100k token, lượng nhiễu đó cộng dồn và nhấn chìm tín hiệu. Differential Transformer (Ye et al., ICLR 2025) khắc phục bằng cách tính differential attention (attention sai phân) như hiệu của hai softmax, trừ đi nền nhiễu chung. DIFF V2 (Microsoft, tháng 1 năm 2026) là bản viết lại cho production stack: độ trễ decode bằng Transformer gốc, không cần kernel tuỳ chỉnh, tương thích FlashAttention. Bài này đi từ V1 tới V2 đầu-cuối, kèm một bản cài đặt đồ chơi của phép trừ mà bạn chạy được bằng Python thuần.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 7, Bài 02 (self-attention), Giai đoạn 7, Bài 15 (các biến thể attention), Giai đoạn 10, Bài 14 (điểm qua kiến trúc)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu chính xác vì sao attention softmax có nền nhiễu (noise floor) và vì sao nó lớn dần theo độ dài ngữ cảnh.
- Suy ra công thức attention sai phân và giải thích vì sao phép trừ triệt tiêu thành phần nhiễu chung mà vẫn giữ tín hiệu.
- Đi qua phần khác biệt V1 → V2: cái gì nhanh hơn, cái gì đơn giản hơn, cái gì ổn định hơn, và vì sao mỗi thay đổi là cần thiết cho pre-training thực tế.
- Tự cài đặt attention sai phân bằng Python thuần và kiểm chứng bằng thực nghiệm tính chất khử nhiễu (noise cancellation) trên một query tổng hợp gồm tín hiệu cộng nhiễu.

## Vấn đề

Attention softmax chuẩn có một tính chất toán học trở thành vấn đề vận hành khi scale lên. Với một query `q`, trọng số attention là `softmax(qK^T / sqrt(d))`. Softmax không bao giờ cho ra số 0 tuyệt đối — mọi token không khớp vẫn nhận một khối xác suất dương. Phần dư đó là nhiễu, và nó tăng theo độ dài ngữ cảnh. Ở 128k token, kể cả khi mỗi token không khớp chỉ nhận 0.001% xác suất thì 127,999 token cộng lại vẫn chiếm khoảng 12% tổng. Mô hình buộc phải học cách lách quanh một nền nhiễu lớn dần theo ngữ cảnh.

Trên thực nghiệm, điều này lộ ra dưới dạng can thiệp giữa các head (đầu) attention: trích dẫn bị ảo giác trong RAG long context (ngữ cảnh dài), lỗi lost-in-the-middle trên tác vụ truy hồi 100k token, và độ chính xác giảm nhẹ trên benchmark needle-in-haystack khi vượt 32k. Bài báo Differential Transformer (arXiv:2410.05258, ICLR 2025) đã đo khoảng cách này: DIFF Transformer đạt perplexity thấp hơn, độ chính xác ngữ cảnh dài cao hơn và ít ảo giác hơn so với baseline cùng kích thước.

DIFF V1 có ba vấn đề khiến nó không lọt vào các pipeline pre-training tiên phong. Value cache của nó phải nạp hai lần mỗi bước decode, nó cần kernel CUDA tuỳ chỉnh nên phá vỡ tương thích FlashAttention, và RMSNorm theo từng head làm bất ổn huấn luyện dài ở quy mô từ 70B trở lên. DIFF V2 (blog Microsoft unilm, 20 tháng 1 năm 2026) sửa cả ba. Bài này đi qua cả hai phiên bản, xây toán tử hiệu, và đo khử nhiễu trên một query đồ chơi.

## Khái niệm

### Nền nhiễu của softmax

Với một query `q` và các key `K = [k_1, ..., k_N]`, trọng số attention là:

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

Không `w_i` nào bằng 0. Nếu `k_i` hoàn toàn không liên quan tới `q`, điểm `q . k_i` không phải 0 — nó dao động quanh 0 với phương sai `||q||^2 / d`. Sau bước normalization (chuẩn hoá) của softmax, mỗi token không liên quan vẫn đóng góp `O(1/N)` vào tổng có trọng số. Tổng đóng góp của các token không liên quan là `O((N-1)/N) = O(1)` — không hề nhỏ.

Cái mô hình muốn là một kiểu top-k cứng: trọng số cao cho token khớp, gần 0 ở mọi chỗ khác. Softmax quá mượt để làm việc đó một cách trực tiếp.

### Ý tưởng sai phân

Tách projection Q và K của mỗi head thành hai: Q = (Q_1, Q_2) và K = (K_1, K_2). Tính hai bản đồ attention:

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

Đầu ra:

```
DiffAttn = (A_1 - lambda * A_2) V
```

Phép trừ triệt tiêu phần nhiễu mà hai bản đồ chia sẻ. Nếu cả hai bản đồ có trọng số gần như đều trên 127k token không liên quan (ở khởi tạo ngẫu nhiên thì đúng là vậy), chúng triệt tiêu nhau. Tín hiệu — trọng số nhọn trên vài token thực sự liên quan — chỉ bị triệt tiêu nếu nó xuất hiện ở cả hai bản đồ với cùng độ lớn, điều sẽ không xảy ra khi mô hình đã huấn luyện.

`lambda` (hệ số lambda) là một scalar học được cho mỗi head, được tham số hoá là `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`. Nó có thể âm. `lambda_init` mặc định là một số dương nhỏ, ví dụ 0.8.

### Vì sao điều này giống khử nhiễu bằng hai micro

Hãy nghĩ tới hai micro ồn cùng thu một giọng nói. Cả hai đều thu giọng người nói cộng nền nhiễu tương quan. Lấy cái này trừ cái kia thì phần nhiễu chung biến mất. Giọng nói vẫn còn vì hai tín hiệu khác nhau đủ về pha hoặc biên độ để không bị triệt tiêu hoàn toàn. `lambda` theo từng head học đúng sự cân bằng đó.

### V1 vs V2: phần khác biệt

V1 giữ số tham số bằng Transformer gốc. Để có hai query mỗi head, nó giảm một nửa chiều head. Cái giá là khả năng biểu diễn của head giảm và — đau hơn — value cache mỗi head cũng giảm một nửa. Decode phải nạp value cache hai lần mỗi bước (một lần cho mỗi nhánh softmax). Kết quả: decode chậm hơn baseline dù số tham số bằng nhau.

V2 nhân đôi số head query và giữ nguyên số head KV (mượn tham số từ up-projection). Chiều head vẫn bằng baseline. Sau phép trừ, chiều dư được chiếu ngược về cho khớp projection O_W của Transformer gốc. Ba thứ xảy ra cùng lúc:

1. Tốc độ decode bằng baseline (KV cache chỉ nạp một lần).
2. FlashAttention chạy không đổi (không cần kernel tuỳ chỉnh).
3. Cường độ số học (arithmetic intensity) khi decode tăng (nhiều tính toán hơn trên mỗi byte nạp từ HBM).

V2 cũng bỏ RMSNorm theo từng head mà V1 dùng để ổn định phép trừ. Ở quy mô pre-training cỡ 70B, RMSNorm đó làm bất ổn giai đoạn cuối huấn luyện. V2 thay bằng một sơ đồ khởi tạo đơn giản hơn, giữ huấn luyện ổn định mà không cần thêm module.

### Khi nào nên dùng

| Tác vụ | Lợi ích |
|----------|---------|
| RAG ngữ cảnh dài (64k+) | Bản đồ attention sạch hơn, ít trích dẫn ảo giác hơn |
| Benchmark needle-in-haystack | Độ chính xác tăng rõ khi vượt 32k |
| QA nhiều tài liệu | Ít can thiệp giữa các tài liệu hơn |
| Code completion ở 8k | Biên lợi ích nhỏ, không đáng đổi kiến trúc |
| Chat ngắn (< 4k) | Gần như không phân biệt được với baseline |

Lợi ích tăng theo độ dài ngữ cảnh. Ở 4k token, nền nhiễu đủ nhỏ nên attention chuẩn vẫn ổn. Ở 128k thì nó đang làm bạn mất điểm.

### Kết hợp với các tham số điều chỉnh khác của 2026

| Tính năng | Tương thích với DIFF V2? |
|---------|------------------------|
| GQA | Có (V2 tăng head Q, không tăng head KV) |
| MLA (DeepSeek) | Về nguyên tắc là có, nhưng chưa có bài báo nào kết hợp cả hai |
| MoE | Có (attention độc lập với khối MLP) |
| RoPE | Có (không đổi) |
| YaRN / long-context scaling | Có (đúng chỗ DIFF giúp nhiều nhất) |
| FlashAttention | V2 thì có (V1 thì không) |
| Speculative decoding | Có (thay đổi attention vô hình với vòng spec-decode) |

```figure
differential-attention
```

## Tự tay làm

`code/main.py` cài đặt attention sai phân bằng Python thuần. Một query đồ chơi có cấu trúc tín hiệu-cộng-nhiễu đã biết cho phép bạn đo trực tiếp tỉ lệ khử nhiễu.

### Bước 1: attention softmax chuẩn

Các phép ma trận thuần stdlib: list lồng list, matmul viết tay, softmax có trừ max để ổn định số học.

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Bước 2: tách Q, K thành hai nửa

Kiểu V1: giảm một nửa chiều head. Kiểu V2: giữ nguyên chiều head và nhân đôi số head. Bản cài đặt đồ chơi dùng V1 cho dễ hiểu — phần toán giống hệt, chỉ khác cách đếm.

### Bước 3: hai nhánh softmax + phép trừ

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

Lưu ý: trọng số đầu ra có thể âm. Không sao — value cache vẫn xử lý được đóng góp có dấu. Projection V phía sau hấp thụ dấu đó.

### Bước 4: đo khử nhiễu

Dựng một chuỗi tổng hợp dài 1024. Đặt token tín hiệu ở một vị trí đã biết, phần còn lại toàn nhiễu. Tính (a) trọng số attention softmax chuẩn tại vị trí tín hiệu và (b) trọng số attention sai phân. Đo tỉ lệ tín hiệu trên nhiễu ở mỗi trường hợp. Attention DIFF cho tỉ lệ tín hiệu trên nhiễu cao hơn một cách ổn định, cỡ 3x-10x tuỳ hai nhánh đã được huấn luyện khác nhau tới đâu.

### Bước 5: hạch toán tham số V1 vs V2

Với cấu hình (hidden=4096, heads=32, d_head=128), in ra:

- Transformer gốc: Q, K, V mỗi cái kích thước `hidden * hidden`, MLP là 4 * hidden.
- DIFF V1: Q, K mỗi cái `hidden * hidden`, V kích thước `hidden * hidden` (không đổi), chiều head bị giảm một nửa bên trong. Cộng thêm tham số `lambda` theo từng head (O(heads * d_head)).
- DIFF V2: Q kích thước `2 * hidden * hidden`, K `hidden * hidden`, V `hidden * hidden`. Chiều dư được chiếu ngược về trước O_W. Cộng thêm đúng bộ tham số `lambda` như trên.

Bản đồ chơi đo chi phí tham số thêm của V2 (khoảng `hidden * hidden` mỗi khối attention) và in ra.

## Áp dụng

Tính tới tháng 4 năm 2026, DIFF V2 chưa có mặt trong mọi inference server production, nhưng vLLM và SGLang đang tích hợp. Trong lúc đó, mẫu thiết kế này xuất hiện ở:

- Các mô hình ngữ cảnh dài production nội bộ của Microsoft.
- Các bản tái lập nghiên cứu trong vài đợt huấn luyện mô hình mở nhắm tới ngữ cảnh từ 256k trở lên.
- Các kiến trúc lai kết hợp attention DIFF với attention cửa sổ trượt trên những lớp xen kẽ.

Khi nào bạn nên dùng nó trong 2026:

- Huấn luyện một mô hình mới từ đầu nhắm tới ngữ cảnh hiệu dụng từ 64k trở lên. Thêm attention sai phân ngay từ đầu; huấn luyện lại sau thì đắt.
- Fine-tune một mô hình ngữ cảnh dài mà lỗi lost-in-the-middle chiếm phần lớn trong eval. Một LoRA trên các projection Q có thể xấp xỉ cấu trúc DIFF.

Khi nào không nên:

- Bạn đang phục vụ một mô hình dense đã huấn luyện xong với hiệu năng ngữ cảnh dài ổn định. Chi phí huấn luyện lại hiếm khi hoàn vốn trên bộ trọng số có sẵn.
- Ngữ cảnh của bạn luôn dưới 16k. Nền nhiễu không đáng kể.

## Đóng gói

Bài này tạo ra `outputs/skill-diff-attention-integrator.md`. Cho trước kiến trúc mô hình, độ dài ngữ cảnh mục tiêu, đặc điểm ảo giác và ngân sách huấn luyện, nó tạo ra một kế hoạch tích hợp attention sai phân vào một lần pre-training mới hoặc một lần fine-tune LoRA.

## Bài tập

1. Chạy `code/main.py`. Kiểm tra rằng tỉ lệ tín hiệu trên nhiễu mà attention sai phân báo cáo cao hơn attention softmax chuẩn trên query tổng hợp. Thay đổi biên độ nhiễu và chỉ ra điểm giao mà từ đó attention chuẩn không còn dùng được.

2. Tính chênh lệch số tham số từ baseline sang DIFF V1 và từ baseline sang DIFF V2 cho mô hình cỡ 7B (hidden=4096, heads=32, d_head=128, 32 lớp). Chỉ rõ thành phần nào tăng tham số và thành phần nào giữ nguyên.

3. Đọc Mục 3 của bài báo DIFF V1 (arXiv:2410.05258) và Mục 2 của blog DIFF V2 trên Hugging Face. Trong hai câu, giải thích vì sao RMSNorm theo từng head của V1 là cần thiết và vì sao V2 bỏ được nó mà không gây phân kỳ huấn luyện.

4. Cài đặt một ablation: tính attention sai phân với `lambda = 0` (chỉ softmax thứ nhất) và `lambda = 1` (trừ đầy đủ). Trên query tổng hợp, đo tỉ lệ tín hiệu trên nhiễu thay đổi ra sao qua cả dải quét. Xác định `lambda` cho tỉ lệ tín hiệu trên nhiễu lớn nhất.

5. Mở rộng bản đồ chơi sang GQA + DIFF V2. Chọn 8 head KV và 32 head Q. Chứng minh kích thước KV cache khớp với mô hình GQA baseline cùng cấu hình (8, 32).

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Differential attention | "Hai softmax trừ nhau" | Tách Q, K thành hai nửa, tính hai bản đồ softmax, lấy bản đồ thứ hai (nhân với lambda) trừ khỏi bản đồ thứ nhất, rồi nhân với V |
| Noise floor | "Phần đuôi khác 0 của softmax" | Trọng số O(1/N) mà softmax đặt lên mọi token không liên quan, cộng lại thành O(1) trên ngữ cảnh dài |
| lambda | "Hệ số của phép trừ" | Tham số vô hướng học được theo từng head, tham số hoá là `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`; có thể âm |
| DIFF V1 | "Bản ICLR 2025" | Differential Transformer gốc; giảm một nửa chiều head để giữ số tham số, cần kernel tuỳ chỉnh, decode chậm hơn |
| DIFF V2 | "Bản sửa của tháng 1 năm 2026" | Nhân đôi head Q và giữ nguyên head KV; tốc độ decode bằng baseline và chạy được với FlashAttention |
| Per-head RMSNorm | "Bộ ổn định của V1" | Phép norm thêm mà V1 áp sau phép trừ; V2 bỏ đi để tránh bất ổn ở cuối quá trình huấn luyện |
| Signal-to-noise ratio | "Attention bị lãng phí bao nhiêu" | Tỉ lệ giữa trọng số ở vị trí tín hiệu thật và trọng số trung bình ở các vị trí không liên quan |
| Lost in the middle | "Kiểu lỗi của ngữ cảnh dài" | Hiện tượng thực nghiệm: độ chính xác truy hồi giảm với tài liệu nằm giữa ngữ cảnh dài — attention DIFF làm giảm hiện tượng này |
| Arithmetic intensity | "Số FLOP trên mỗi byte nạp" | Tỉ lệ V2 tăng lúc decode nhờ nhân đôi query trên mỗi lần nạp KV; quan trọng với decode bị giới hạn bởi bộ nhớ |

## Đọc thêm

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — bài báo gốc với lý thuyết khử nhiễu và các ablation ngữ cảnh dài
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — bản viết lại cho production stack, decode bằng baseline, tương thích FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — phân tích lý thuyết vì sao phép trừ khôi phục cấu trúc attention đã pretrain
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — biến thể chia sẻ tham số
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — Transformer gốc mà DIFF trừ đi
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — benchmark ngữ cảnh dài mà attention DIFF nhắm tới

---

[← Bài trước: Giải mã suy đoán và EAGLE-3](../../15-speculative-decoding-eagle3/docs/vi.md) · [Bài sau: Native Sparse Attention (DeepSeek NSA) →](../../17-native-sparse-attention/docs/vi.md)
