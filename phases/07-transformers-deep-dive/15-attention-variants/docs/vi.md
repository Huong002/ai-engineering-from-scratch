# Các biến thể attention — Cửa sổ trượt, Thưa, Vi phân

> Full attention giống một vòng tròn. Mỗi token nhìn thấy mọi token, và bộ nhớ phải trả giá. Bốn biến thể bẻ cong hình dạng vòng tròn đó và lấy lại một nửa chi phí.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 02 (Self-attention), Giai đoạn 7, Bài 03 (Multi-head), Giai đoạn 7, Bài 12 (KV cache / Flash attention)
**Time:** ~60 phút

## Vấn đề

Full attention tốn `O(N²)` bộ nhớ và `O(N²)` tính toán theo độ dài chuỗi. Với Llama 3 70B ngữ cảnh 128K, đó là 16 tỷ phần tử attention mỗi lớp, nhân với 80 lớp. Flash Attention (Bài 12) che giấu bộ nhớ activation `O(N²)` nhưng không đổi chi phí số học — mỗi token vẫn attend tới mọi token khác.

Ba nhóm biến thể thay đổi cấu trúc liên kết của chính ma trận attention:

1. **Sliding window attention (SWA, attention cửa sổ trượt).** Mỗi token attend tới một cửa sổ lân cận cố định, không phải toàn bộ tiền tố. Bộ nhớ và tính toán giảm xuống `O(N · W)` với `W` là cửa sổ. Gemma 2/3, các lớp đầu của Mistral 7B, Phi-3-Long.
2. **Sparse / block attention (attention thưa / theo khối).** Chỉ những cặp `(i, j)` được chọn mới được tính điểm; phần còn lại bị ép trọng số bằng 0. Longformer, BigBird, OpenAI sparse transformer.
3. **Differential attention.** Tính hai attention map với hai phép chiếu Q/K riêng, rồi trừ cái này cho cái kia. Diệt "attention sink" — chỗ rò trọng số vào vài token đầu tiên. DIFF Transformer của Microsoft (2024).

Các cách này cùng tồn tại. Một mô hình tiên phong năm 2026 thường trộn chúng: phần lớn lớp là SWA-1024, cứ lớp thứ năm là full attention toàn cục, và một vài head vi phân dọn lại phần truy hồi. Tỉ lệ 5:1 SWA-trên-toàn cục của Gemma 3 hiện là mặc định chuẩn mực.

## Khái niệm

### Sliding Window Attention (SWA)

Mỗi query ở vị trí `i` chỉ attend tới các vị trí trong `[i - W, i]` (SWA nhân quả) hoặc `[i - W/2, i + W/2]` (hai chiều). Các token ngoài cửa sổ nhận `-inf` trong ma trận điểm.

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

Với `N = 8192` và `W = 1024`, ma trận điểm có 1024 × 8192 hàng khác 0 theo kỳ vọng — giảm 8×.

**KV cache co lại nhờ SWA.** Mỗi lớp chỉ cần giữ `W` token cuối của K và V. Với cấu hình kiểu Gemma-3 (cửa sổ 1024, ngữ cảnh 128K), KV cache giảm 128×.

**Cái giá về chất lượng.** Transformer chỉ dùng SWA gặp khó với truy hồi tầm xa. Cách sửa: xen kẽ lớp SWA với lớp full attention. Gemma 3 dùng tỉ lệ 5:1 SWA:toàn cục. Mistral 7B dùng một chồng causal-SWA nơi thông tin "chảy về phía trước" qua các cửa sổ chồng lấn — mỗi lớp mở rộng trường tiếp nhận hiệu dụng thêm `W`, và sau `L` lớp mô hình có thể attend ngược `L × W` token.

### Sparse / Block Attention

Chọn trước một mẫu thưa `N × N`. Ba dạng kinh điển:

- **Local + strided (OpenAI sparse transformer).** Attend tới `W` token cuối cùng cộng với mỗi token thứ `stride` trước đó. Bắt được cả cục bộ lẫn tầm xa với `O(N · sqrt(N))` tính toán.
- **Longformer / BigBird.** Cửa sổ cục bộ + một tập nhỏ token toàn cục (ví dụ `[CLS]`) attend tới mọi người và được mọi người attend + các liên kết thưa ngẫu nhiên. Thực nghiệm cho ngữ cảnh gấp 2× ở chất lượng tương đương.
- **Native Sparse Attention (DeepSeek, 2025).** Học xem khối `(Q, K)` nào quan trọng; bỏ qua các khối bằng 0 ở mức kernel (nhân tính toán). Tương thích FlashAttention.

Sparse attention là câu chuyện kỹ thuật kernel. Toán thì đơn giản (mask ma trận điểm); lợi ích đến từ việc không bao giờ nạp các phần tử bằng 0 vào SRAM. FlashAttention-3 và API FlexAttention năm 2026 biến các mẫu thưa tùy biến thành công dân hạng nhất trong PyTorch.

### Differential Attention (DIFF Transformer, 2024)

Attention thường có vấn đề "attention sink": softmax ép mỗi hàng cộng lại bằng 1, nên những token không muốn attend vào đâu cả sẽ dồn trọng số vào token đầu tiên (hoặc vài token đầu). Điều này lấy mất dung lượng lẽ ra dành cho nội dung thật.

Differential attention sửa chuyện đó bằng cách tính **hai** attention map rồi trừ nhau:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

trong đó `λ` là một scalar học được (thường 0.5–0.8). A1 nắm trọng số nội dung thật; A2 nắm phần sink. Phép trừ triệt tiêu sink và phân bổ lại trọng số cho các token liên quan.

Kết quả công bố (Microsoft 2024): perplexity thấp hơn 5–10%, ngữ cảnh hiệu dụng dài hơn 1.5–2× ở cùng độ dài huấn luyện, truy hồi needle-in-haystack sắc nét hơn.

### So sánh các biến thể

| Biến thể | Chi phí tính toán | KV cache | Chất lượng so với full | Dùng trong production |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | O(N) mỗi lớp | mức cơ sở | lớp mặc định của mọi mô hình |
| SWA (cửa sổ 1024) | O(N·W) | O(W) mỗi lớp | -0.1 ppl, tốt khi có lớp toàn cục | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | hỗn hợp | tương đương SWA | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | O(N) xấp xỉ | hỗn hợp | ngang full ở ngữ cảnh 2× | BERT ngữ cảnh dài thời đầu |
| Native Sparse (DeepSeek-V3.2) | O(N · tỉ lệ hoạt động) | O(N) | trong khoảng 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5 đến -10% ppl | DIFF Transformer, các mô hình đầu 2026 |

```figure
gqa-kv-sharing
```

## Tự tay làm

Xem `code/main.py`. Ta cài một bộ so sánh mask nhân quả, đặt full, SWA, local+strided và differential attention cạnh nhau trên một chuỗi đồ chơi.

### Bước 1: mask nhân quả đầy đủ (cơ sở)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Bản cơ sở từ Bài 07. Tam giác dưới; trọng số bằng 0 phía trên đường chéo.

### Bước 2: mask nhân quả cửa sổ trượt

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

Một tham số — `window`. Với `window >= n`, bạn thu lại full attention nhân quả. Với `window = 1`, mỗi token chỉ attend tới chính nó.

### Bước 3: mask thưa local + strided

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Cửa sổ cục bộ đặc, cộng thêm mỗi token thứ `stride` ngược về đầu chuỗi. Trường tiếp nhận lớn dần theo bước log khi thêm lớp.

### Bước 4: differential attention

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

Hai lượt attention, trừ nhau với một hệ số trộn học được. Trong code ta so sánh heatmap attention-sink của đơn và vi phân để xem sink sụp xuống.

### Bước 5: kích thước KV cache

In kích thước cache mỗi lớp ở `N = 131072` cho từng biến thể. SWA và các biến thể thưa giảm 10–100×. Differential thì gấp đôi. Hãy trả hoá đơn bộ nhớ một cách có ý thức.

## Áp dụng

Các mẫu production năm 2026:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

FlexAttention trong PyTorch 2.5+ nhận một hàm mask:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

Đoạn này biên dịch thành một kernel Triton tùy biến. Đạt trong khoảng 10% tốc độ FlashAttention-3 với các mẫu phổ biến, và hàm mask chỉ là một callable Python.

**Khi nào chọn cái nào:**

- **Full attention thuần** — mọi lớp cho tới ngữ cảnh ~16K, hoặc khi chất lượng truy hồi là quan trọng nhất.
- **SWA + trộn toàn cục** — ngữ cảnh dài (>32K), huấn luyện và suy luận bị nghẽn bộ nhớ. Mặc định năm 2026 cho trên 32K.
- **Sparse block attention** — kernel riêng, mẫu riêng. Dành cho khối lượng công việc chuyên biệt (truy hồi, âm thanh).
- **Differential attention** — mọi khối lượng công việc nơi nhiễm attention-sink gây hại (RAG ngữ cảnh dài, needle-in-haystack).

## Đóng gói

Xem `outputs/skill-attention-variant-picker.md`. Skill này chọn một cấu trúc attention cho mô hình mới dựa trên độ dài ngữ cảnh mục tiêu, yêu cầu truy hồi, và hồ sơ tính toán huấn luyện/suy luận.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Kiểm chứng SWA với `window=4` đặt mọi thứ ngoài 4 token cuối của mỗi hàng về 0. Kiểm chứng `window=n` tái tạo full attention nhân quả giống hệt từng bit.
2. **Trung bình.** Cài SWA nhân quả với `window=1024` lên trên capstone của Bài 07. Huấn luyện 1.000 bước trên tinyshakespeare. Val loss tệ đi bao nhiêu so với full attention? Bộ nhớ đỉnh giảm bao nhiêu?
3. **Khó.** Cài kiểu trộn lớp 5:1 kiểu Gemma-3 (5 SWA, 1 toàn cục) vào mô hình capstone. So sánh loss, bộ nhớ và chất lượng sinh với các baseline thuần-SWA và thuần-toàn cục ở cùng số tham số.
4. **Khó.** Cài differential attention với `λ` học được cho từng head. Huấn luyện trên một tác vụ truy hồi tổng hợp (một needle, 2.000 distractor). Đo độ chính xác truy hồi so với baseline attention đơn ở cùng số tham số.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | "Attention cục bộ" | Mỗi query attend tới `W` token cuối của nó; KV cache co lại còn `O(W)`. |
| Effective receptive field | "Mô hình nhìn ngược được bao xa" | Trong một chồng SWA `L` lớp với cửa sổ `W`, tới `L × W` token. |
| Longformer / BigBird | "Local + global + random" | Mẫu thưa với vài token toàn cục luôn attend; cách tiếp cận ngữ cảnh dài thời đầu. |
| Native Sparse Attention | "Chiêu kernel của DeepSeek" | Học độ thưa ở mức khối; bỏ qua khối bằng 0 ở mức kernel mà vẫn giữ chất lượng. |
| Differential attention | "Hai map, một phép trừ" | DIFF Transformer: trừ map thứ hai nhân `λ` học được khỏi map thứ nhất để triệt tiêu attention sink. |
| Attention sink | "Trọng số rò về token 0" | Chuẩn hoá softmax ép các hàng cộng lại bằng 1; query không mang thông tin dồn trọng số vào vị trí 0. |
| FlexAttention | "Mask là Python" | API PyTorch 2.5+ biên dịch các hàm mask tuỳ ý thành kernel hình dạng FlashAttention. |
| Layer type mix | "5:1 SWA-trên-toàn cục" | Xen kẽ lớp attention thưa và full trong một chồng để giữ chất lượng với ít bộ nhớ hơn. |

## Đọc thêm

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — bài báo chuẩn mực về cửa sổ trượt + token toàn cục.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — local + global + random.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — mẫu local+strided của OpenAI.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — kiểu trộn 1:1 SWA:toàn cục.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — kiểu trộn 5:1 với window=1024, nay là mặc định chuẩn mực.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — bài báo DIFF Transformer.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — attention thưa học được của DeepSeek-V3.2.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — tài liệu API cho kiểu mask-là-callable trong phần Áp dụng.

---

[← Bài trước: Tự xây dựng Transformer từ đầu](../../14-build-a-transformer-capstone/docs/vi.md) · [Bài sau: Speculative decoding (giải mã suy đoán) →](../../16-speculative-decoding/docs/vi.md)
