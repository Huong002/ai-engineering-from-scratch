# Positional Encoding — Sinusoidal, RoPE, ALiBi

> Attention có tính permutation invariance (bất biến hoán vị). "The cat sat on the mat" và "mat the on sat cat the" cho ra cùng một output nếu không có tín hiệu positional encoding (mã hoá vị trí). Ba thuật toán khắc phục điều này — mỗi thuật toán đặt cược vào một cách hiểu khác nhau về "vị trí".

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 02 (Self-attention), Giai đoạn 7, Bài 03 (Multi-head attention)
**Time:** ~45 phút

## Vấn đề

Scaled dot-product attention không phân biệt thứ tự. Ma trận attention `softmax(Q K^T / √d) V` được tính từ độ tương đồng của từng cặp. Xáo trộn các hàng của `X`, bạn nhận lại các hàng output bị xáo trộn y hệt. Không có gì bên trong attention quan tâm tới vị trí.

Đó không phải lỗi trong một mô hình bag-of-words. Với ngôn ngữ, code, âm thanh, video — bất cứ thứ gì mà thứ tự mang ý nghĩa — đó là lỗi chí mạng.

Cách khắc phục là đưa vị trí vào embedding theo một cách nào đó. Ba thời kỳ câu trả lời:

1. **Absolute sinusoidal** (hình sin tuyệt đối, Vaswani 2017). Cộng `sin/cos` của vị trí vào embedding. Đơn giản, không cần học, extrapolation (ngoại suy) kém khi vượt quá độ dài đã huấn luyện.
2. **RoPE — Rotary Position Embeddings** (embedding vị trí dạng xoay, Su 2021). Xoay các vector Q và K một góc tỉ lệ với vị trí. Mã hoá relative position (vị trí tương đối) trực tiếp trong tích vô hướng. Thống trị vào năm 2026.
3. **ALiBi — Attention with Linear Biases** (Press 2022). Bỏ qua embedding hoàn toàn; cộng một hình phạt tuyến tính theo từng head vào điểm attention dựa trên khoảng cách. Ngoại suy độ dài rất tốt.

Tính đến 2026, gần như mọi mô hình mở tiên phong đều dùng RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. Một số ít mô hình long-context dùng ALiBi hoặc các biến thể hiện đại của nó. Absolute sinusoidal chỉ còn là lịch sử.

## Khái niệm

![Sinusoidal tuyệt đối so với phép xoay RoPE so với độ lệch khoảng cách của ALiBi](../assets/positional-encoding.svg)

### Sinusoidal tuyệt đối

Tính trước một ma trận cố định `PE` có shape `(max_len, d_model)`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

Sau đó `X' = X + PE[:N]` trước khi vào attention. Mỗi chiều là một dao động hình sin ở một frequency (tần số) khác nhau. Mô hình học cách đọc vị trí từ kiểu pha. Hỏng khi vượt `max_len`: không gì cho mô hình biết điều gì xảy ra ở vị trí 2048 khi nó chỉ từng thấy các vị trí 0–2047.

### RoPE

Xoay các vector Q và K (không phải embedding). Với một cặp chiều `(2i, 2i+1)`:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

Áp dụng cùng phép xoay đó cho các key với vị trí `pos_k`. Tích vô hướng `q'_m · k'_n` trở thành một hàm chỉ của `(m - n)`. Nghĩa là: **điểm attention chỉ phụ thuộc vào khoảng cách tương đối**, dù phép xoay được tính từ vị trí tuyệt đối. Một chiêu rất đẹp.

Mở rộng RoPE: có thể scale `base` (NTK-aware, YaRN, LongRoPE) để ngoại suy sang context dài hơn mà không cần huấn luyện lại. Llama 3 mở rộng từ context 8K lên 128K theo cách này.

### ALiBi

Bỏ qua chiêu embedding. Lệch điểm attention trực tiếp:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

Trong đó `m_h` là độ dốc riêng cho từng head (ví dụ `1 / 2^(8·h/H)`). Token gần được nâng, token xa bị phạt. Không tốn chi phí lúc huấn luyện. Bài báo cho thấy ngoại suy độ dài của nó tốt hơn sinusoidal và ngang RoPE ở độ dài nó từng được huấn luyện.

### Chọn gì vào năm 2026

| Biến thể | Khả năng ngoại suy | Chi phí huấn luyện | Được dùng bởi |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | kém | miễn phí | transformer gốc, BERT thời đầu |
| Learned absolute | không có | rất nhỏ | GPT-2, GPT-3 |
| RoPE | tốt nếu có scaling | miễn phí | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | xuất sắc | giai đoạn fine-tune | Qwen2-1M, Llama 3.1 128K |
| ALiBi | xuất sắc | miễn phí | BLOOM, MPT, Baichuan |

RoPE thắng vì nó lắp vào attention mà không đổi kiến trúc, mã hoá vị trí tương đối, và siêu tham số `base` của nó cho một núm vặn gọn gàng để fine-tune cho long-context.

```figure
rope-explorer
```

## Tự tay làm

### Bước 1: mã hoá sinusoidal

Xem `code/main.py`. Một phép tính 4 dòng:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Cộng ma trận này vào embedding trước lớp attention đầu tiên.

### Bước 2: áp RoPE lên Q, K

RoPE thao tác tại chỗ trên Q và K. Với từng cặp chiều:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Điểm mấu chốt: áp cùng một hàm cho Q ở vị trí `m` và K ở vị trí `n`. Tích vô hướng của chúng nhận thêm một nhân tử `cos((m-n)·θ_i)` trên mỗi cặp toạ độ. Attention học được vị trí tương đối mà không tốn gì.

### Bước 3: độ dốc và bias của ALiBi

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Cộng `bias[h]` vào ma trận điểm attention `(seq_len, seq_len)` của head `h`, rồi softmax.

### Bước 4: kiểm chứng tính chất khoảng cách tương đối của RoPE

Chọn hai vector ngẫu nhiên `a, b`. Xoay theo `(pos_a, pos_b)`. Rồi xoay theo `(pos_a + k, pos_b + k)`. Hai tích vô hướng phải khớp nhau trong sai số dấu phẩy động. Tính chất đó chính là toàn bộ ý nghĩa của RoPE — nó bất biến với độ dịch tuyệt đối, chỉ khoảng cách tương đối là quan trọng.

## Áp dụng

PyTorch 2.5+ có sẵn các tiện ích RoPE trong `torch.nn.functional`. Phần lớn code production dùng `flash_attn` hoặc `xformers`, nơi RoPE được áp ngay bên trong kernel attention.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Mẹo long-context năm 2026:**

- **NTK-aware interpolation.** Scale lại `base` thành `base * (scale_factor)^(d/(d-2))` khi mở rộng từ 4K lên 16K+.
- **YaRN.** Cách nội suy khéo hơn, giữ được entropy attention trên các context dài. Llama 3.1 128K dùng nó.
- **LongRoPE.** Phương pháp năm 2024 của Microsoft dùng tìm kiếm tiến hoá để chọn hệ số scale cho từng chiều. Phi-3-Long dùng nó.
- **Position interpolation + fine-tuning.** Chỉ cần thu nhỏ vị trí theo hệ số mở rộng rồi fine-tune trên 1–5B token. Hiệu quả bất ngờ.

## Đóng gói

Xem `outputs/skill-positional-encoding-picker.md`. Skill này chọn một chiến lược mã hoá cho mô hình mới, dựa trên độ dài context mục tiêu, nhu cầu ngoại suy và ngân sách huấn luyện.

## Bài tập

1. **Dễ.** Vẽ ma trận `PE` sinusoidal thành heatmap với `max_len=512, d=128`. Xác nhận kiểu "các sọc rộng dần khi chỉ số chiều tăng".
2. **Trung bình.** Cài đặt NTK-aware RoPE scaling. Huấn luyện một LM nhỏ trên các chuỗi dài 256, rồi test trên độ dài 1024 khi có và khi không có scaling. Đo perplexity.
3. **Khó.** Cài đặt ALiBi và RoPE trong cùng một module attention. Huấn luyện một transformer 4 lớp trên bài toán copy với chuỗi dài 512. Ngoại suy lên 2048 lúc test. So sánh mức suy giảm.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Positional encoding | "Cho attention biết về thứ tự" | Bất kỳ tín hiệu nào được cộng vào embedding hoặc attention để mã hoá vị trí. |
| Sinusoidal | "Cái nguyên bản ấy" | `sin/cos` ở các tần số cấp số nhân được cộng vào embedding; không ngoại suy được. |
| RoPE | "Embedding dạng xoay" | Xoay Q, K một góc phụ thuộc vị trí; tích vô hướng mã hoá khoảng cách tương đối. |
| ALiBi | "Chiêu bias tuyến tính" | Cộng `-m·\|i-j\|` vào điểm attention; không cần embedding, ngoại suy rất tốt. |
| base | "Núm vặn của RoPE" | Hệ số scale tần số trong RoPE; tăng nó để mở rộng context lúc inference. |
| NTK-aware | "Một chiêu scale RoPE" | Scale lại `base` để các chiều tần số cao không bị nén khi context mở rộng. |
| YaRN | "Cái xịn hơn" | Nội suy + ngoại suy theo từng chiều, giữ được entropy attention. |
| Extrapolation | "Chạy được ngoài độ dài đã học" | Chiến lược vị trí có cho output đúng vượt `max_len` đã thấy khi huấn luyện không? |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — sinusoidal nguyên bản.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — bài báo RoPE.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — RoPE scaling tiên tiến nhất.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — bài báo long-context của Meta cho Llama 2.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — phương pháp của Microsoft được Phi-3-Long dùng và được nhắc trong mục Áp dụng.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — cài đặt chuẩn production cho mọi sơ đồ RoPE scaling (default, linear, dynamic, YaRN, LongRoPE, Llama-3).

---

[← Bài trước: Multi-Head Attention](../../03-multi-head-attention/docs/vi.md) · [Bài sau: Transformer đầy đủ →](../../05-full-transformer/docs/vi.md)
