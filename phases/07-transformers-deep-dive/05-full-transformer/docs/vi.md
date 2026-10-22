# Transformer đầy đủ — Encoder + Decoder

> Attention là ngôi sao. Mọi thứ còn lại — residual (kết nối dư), normalization (chuẩn hoá), feed-forward (lan truyền xuôi), cross-attention — chỉ là giàn giáo giúp bạn xếp chồng nó thật sâu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 02 (Self-attention), Giai đoạn 7, Bài 03 (Multi-head attention), Giai đoạn 7, Bài 04 (Positional encoding)
**Time:** ~75 phút

## Vấn đề

Một lớp attention đơn lẻ là bộ trích xuất đặc trưng, không phải mô hình. Mỗi lớp chỉ một phép matmul là không đủ năng lực cho ngôn ngữ. Bạn cần độ sâu — và độ sâu sẽ sụp nếu thiếu hệ thống đường ống đúng.

Bài báo Vaswani năm 2017 đóng gói sáu quyết định thiết kế, biến một lớp attention thành một block xếp chồng được. Mọi transformer về sau — chỉ encoder (BERT), chỉ decoder (GPT), encoder-decoder (T5) — đều thừa hưởng cùng bộ khung. Đến 2026 các block đã được tinh chỉnh (RMSNorm, SwiGLU, pre-norm, RoPE) nhưng bộ khung vẫn y nguyên.

Bài này chính là bộ khung đó. Các bài sau chuyên biệt hoá nó — 06 cho encoder, 07 cho decoder, 08 cho encoder-decoder.

## Khái niệm

![Bên trong block encoder và decoder, cách nối với nhau](../assets/full-transformer.svg)

### Sáu thành phần

1. **Embedding + tín hiệu vị trí.** Token → vector. Vị trí được đưa vào qua RoPE (hiện đại) hoặc sinusoidal (cổ điển).
2. **Self-attention.** Mọi vị trí chú ý đến mọi vị trí khác. Trong decoder thì có mask.
3. **Feed-forward network (FFN).** MLP hai lớp áp dụng theo từng vị trí: `W_2 · activation(W_1 · x)`. Tỷ lệ mở rộng mặc định 4×.
4. **Residual connection.** `x + sublayer(x)`. Không có nó, gradient biến mất sau khoảng 6 lớp.
5. **Layer normalization.** `LayerNorm` hoặc `RMSNorm` (hiện đại). Ổn định luồng residual.
6. **Cross-attention (chỉ decoder).** Query đến từ decoder, key và value đến từ output của encoder.

Xem một vector chảy qua một block: attention trộn thông tin giữa các vị trí, residual mang nó đi tiếp, FFN biến đổi nó, và norm giữ luồng ổn định.

```figure
transformer-block
```

### Block encoder (BERT và encoder của T5 dùng)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder là hai chiều. Không có mask. Mọi vị trí đều thấy mọi vị trí.

### Block decoder (GPT và decoder của T5 dùng)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Mỗi block decoder có ba lớp con (sublayer). Cái ở giữa — cross-attention — là nơi duy nhất thông tin chảy từ encoder sang decoder. Trong kiến trúc thuần chỉ decoder (GPT), cross-attention bị bỏ đi và bạn chỉ còn masked self-attention + FFN.

### Pre-norm và post-norm

Bài báo gốc: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. Post-norm (chuẩn hoá sau) mất ưu thế quanh 2019 — khó huấn luyện sâu nếu không warmup kỹ. Pre-norm (chuẩn hoá trước, `LN` *trước* lớp con) là mặc định 2026: Llama, Qwen, GPT-3+, Mistral đều dùng nó.

### Block hiện đại hoá kiểu 2026

Vaswani 2017 dùng LayerNorm + ReLU. Các stack hiện đại thay cả hai. Block trong production thực tế trông thế này:

| Thành phần | 2017 | 2026 |
|-----------|------|------|
| Chuẩn hoá | LayerNorm | RMSNorm |
| Kích hoạt FFN | ReLU | SwiGLU |
| Mở rộng FFN | 4× | 2.6× (SwiGLU dùng ba ma trận, tổng tham số tương đương) |
| Vị trí | Sinusoidal tuyệt đối | RoPE |
| Attention | MHA đầy đủ | GQA (hoặc MLA) |
| Số hạng bias | Có | Không |

RMSNorm bỏ bước trừ mean của LayerNorm (bớt một phép trừ), tiết kiệm tính toán và theo thực nghiệm thì ít nhất cũng ổn định ngang. SwiGLU (`Swish(W1 x) ⊙ W3 x`) vượt ReLU/GELU FFN khoảng 0.5 điểm ppl một cách nhất quán trong các bài báo Llama, PaLM và Qwen.

### Đếm tham số

Với một block có `d_model = d` và tỷ lệ mở rộng FFN `r`:

- MHA: `4 · d²` (các projection Q, K, V, O)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Norm: không đáng kể

Với `d = 4096, r = 2.6, layers = 32` (xấp xỉ Llama 3 8B), tổng: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B` (cộng thêm embedding và head). Khớp với các con số đã công bố.

## Tự tay làm

### Bước 1: các khối xây dựng

Dùng lớp `Matrix` nhỏ xíu từ Bài 03 (chép vào file này để bài học độc lập):

- `layer_norm(x, eps=1e-5)` — trừ mean, chia cho std.
- `rms_norm(x, eps=1e-6)` — chia cho RMS. Không trừ mean.
- `gelu(x)` và `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)` và `decoder_block(x, enc_out, params)`.

Xem `code/main.py` để thấy toàn bộ cách nối.

### Bước 2: nối một encoder 2 lớp và một decoder 2 lớp

Xếp chồng chúng. Truyền output của encoder vào cross-attention của mọi decoder. Thêm một LN cuối cùng trước projection output.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Bước 3: chạy forward trên một ví dụ đồ chơi

Đưa một nguồn 6 token và một đích 5 token qua. Kiểm tra shape output là `(5, vocab)`. Không huấn luyện — bài này nói về kiến trúc, không phải loss.

### Bước 4: thay bằng RMSNorm + SwiGLU

Thay LayerNorm và ReLU-FFN bằng RMSNorm và SwiGLU. Xác nhận shape vẫn khớp. Đây là bước hiện đại hoá 2026 chỉ với một lần thay hàm.

## Áp dụng

Các bản tham chiếu PyTorch/TF: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`. Nhưng phần lớn code production 2026 tự viết block vì:

- Flash Attention được gọi bên trong attention, không qua `nn.MultiheadAttention`.
- GQA / MLA không có trong bản tham chiếu của stdlib.
- RoPE, RMSNorm, SwiGLU không phải mặc định của PyTorch.

HF `transformers` có những block tham chiếu gọn gàng mà bạn nên đọc: `modeling_llama.py` là block chỉ decoder chuẩn mực của 2026. Nó dài khoảng 500 dòng và đáng để đọc qua một lần.

**Encoder vs decoder vs encoder-decoder — khi nào chọn cái nào:**

| Nhu cầu | Chọn | Ví dụ |
|------|------|---------|
| Phân loại, embedding, QA trên văn bản | Chỉ encoder | BERT, DeBERTa, ModernBERT |
| Sinh văn bản, chat, code, suy luận | Chỉ decoder | GPT, Llama, Claude, Qwen |
| Đầu vào có cấu trúc → đầu ra có cấu trúc (dịch, tóm tắt) | Encoder-decoder | T5, BART, Whisper |

Chỉ decoder thắng ở lĩnh vực ngôn ngữ vì nó mở rộng quy mô sạch nhất và xử lý được cả hiểu lẫn sinh. Encoder-decoder vẫn tốt nhất khi đầu vào có danh tính "chuỗi nguồn" rõ ràng (dịch, nhận dạng giọng nói, các tác vụ có cấu trúc).

## Đóng gói

Xem `outputs/skill-transformer-block-reviewer.md`. Skill này đánh giá một implementation transformer block mới so với các mặc định 2026 và chỉ ra những phần còn thiếu (pre-norm, RoPE, RMSNorm, GQA, tỷ lệ mở rộng FFN).

## Bài tập

1. **Dễ.** Đếm số tham số trong `encoder_block` của bạn với `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Kiểm chứng bằng cách implement block rồi dùng `sum(p.numel() for p in block.parameters())`.
2. **Trung bình.** Chuyển từ post-norm sang pre-norm. Khởi tạo cả hai rồi đo norm của activation sau 12 lớp xếp chồng trên input ngẫu nhiên. Activation của post-norm sẽ nổ; của pre-norm sẽ giữ mức có chặn.
3. **Khó.** Implement một encoder-decoder 4 lớp cho tác vụ copy đồ chơi (copy `x` theo thứ tự đảo). Huấn luyện 100 bước. Báo loss. Thay bằng RMSNorm + SwiGLU + RoPE — loss có giảm không?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Block | "Một lớp transformer" | Chồng norm + attention + norm + FFN, bọc trong residual connection. |
| Residual | "Kết nối tắt" | Output `x + f(x)`; giúp gradient chảy qua các stack sâu. |
| Pre-norm | "Chuẩn hoá trước, không phải sau" | Hiện đại: `x + sublayer(LN(x))`. Huấn luyện sâu hơn mà không cần warmup rườm rà. |
| RMSNorm | "LayerNorm không có mean" | Chia cho RMS; bớt một phép, độ ổn định thực nghiệm như nhau. |
| SwiGLU | "FFN mà ai cũng chuyển sang" | `Swish(W1 x) ⊙ W3 x → W2`. Hơn ReLU/GELU về ppl trên LM. |
| Cross-attention | "Cách decoder nhìn encoder" | MHA với Q từ decoder, K/V từ output của encoder. |
| FFN expansion | "Độ rộng của MLP ở giữa" | Tỷ lệ hidden-size trên d_model, thường là 4 (LayerNorm) hoặc 2.6 (SwiGLU). |
| Bias-free | "Bỏ các hạng +b" | Các stack hiện đại bỏ bias trong lớp linear; ppl tốt hơn chút, mô hình nhỏ hơn. |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — đặc tả block gốc.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — vì sao pre-norm thắng post-norm ở độ sâu lớn.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — bài báo SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — block chỉ decoder chuẩn mực của 2026.

---

[← Bài trước: Positional Encoding](../../04-positional-encoding/docs/vi.md) · [Bài sau: BERT →](../../06-bert-masked-language-modeling/docs/vi.md)
