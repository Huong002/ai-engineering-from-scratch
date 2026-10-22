# Tự xây dựng Transformer từ đầu — Đồ án tổng kết

> Mười ba bài học. Một mô hình. Không có đường tắt.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 01 đến 13. Đừng bỏ qua.
**Time:** ~120 phút

## Vấn đề

Bạn đã đọc đủ các paper. Bạn đã tự viết attention, tách multi-head, positional encoding, khối encoder và decoder, loss của BERT và GPT, MoE, KV cache. Giờ là lúc làm chúng chạy cùng nhau trên một bài toán thật.

Đồ án tổng kết (capstone): huấn luyện end-to-end một transformer nhỏ chỉ có decoder trên bài toán mô hình ngôn ngữ ở mức ký tự. Nó đọc Shakespeare. Nó sinh ra Shakespeare mới. Nó đủ nhỏ để huấn luyện trên laptop trong dưới 10 phút. Nó đúng đủ để khi bạn thay bằng bộ dữ liệu lớn hơn và huấn luyện lâu hơn, bạn có một LM thật.

Đây là "nanoGPT" của khoá học. Nó không phải ý tưởng gốc — tutorial nanoGPT năm 2023 của Karpathy là bản tham chiếu mà sinh viên nào cũng viết ít nhất một lần. Chúng ta lấy đúng hình dạng đó rồi lắp lại theo những gì đã học.

## Khái niệm

![Sơ đồ khối transformer xây từ đầu](../assets/capstone.svg)

Kiến trúc, kèm chú thích:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### Những gì chúng ta giao

- `GPTConfig` — một chỗ duy nhất để cấu hình mọi hyperparameter (siêu tham số).
- `MultiHeadAttention` — causal, chạy theo batch, có đường đi kiểu Flash tuỳ chọn (qua `scaled_dot_product_attention` của PyTorch).
- `SwiGLUFFN` — FFN kiểu hiện đại.
- `Block` — pre-norm, attention + FFN bọc trong residual.
- `GPT` — embedding, các block xếp chồng, LM head, generate().
- Vòng huấn luyện (training loop) với AdamW, cosine LR, gradient clipping.
- Tokenizer mức ký tự trên văn bản Shakespeare.

### Những gì không có

- RoPE — chỉ làm trên ý tưởng ở Bài 04. Ở đây chúng ta dùng positional embedding học được cho đơn giản. Phần bài tập yêu cầu bạn thay bằng RoPE.
- KV cache khi sinh — mỗi bước sinh tính lại attention trên toàn bộ tiền tố. Chậm hơn nhưng đơn giản hơn. Phần bài tập yêu cầu bạn thêm KV cache.
- Flash Attention — PyTorch 2.0+ tự điều phối nếu đầu vào khớp; chúng ta dùng `F.scaled_dot_product_attention`.
- MoE — mỗi block chỉ một FFN. Bạn đã thấy MoE ở Bài 11.

### Chỉ số mục tiêu

Trên một laptop Mac M2, với GPT 4 lớp, 4 head, d_model=128, huấn luyện 2.000 bước trên `tinyshakespeare.txt`:

- Loss huấn luyện hội tụ từ ~4.2 (ngẫu nhiên) xuống ~1.5 trong khoảng 6 phút.
- Kết quả lấy mẫu trông có dáng Shakespeare: từ cổ, ngắt dòng, tên riêng như "ROMEO:" xuất hiện.
- Val loss (10% cuối văn bản, giữ riêng ra) bám sát loss huấn luyện; không có overfitting (quá khớp) ở kích thước và ngân sách này.

```figure
n5-block-stack
```

## Tự tay làm

Bài này dùng PyTorch. Cài `torch` (bản CPU là đủ). Xem `code/main.py`. Script lo các việc:

- Tải `tinyshakespeare.txt` nếu thiếu (hoặc đọc bản sao cục bộ).
- Tokenizer ký tự ở mức byte.
- Chia train/val theo tỉ lệ 90/10.
- Vòng huấn luyện với bf16 autocast trên phần cứng hỗ trợ.
- Lấy mẫu sau khi huấn luyện xong.

### Bước 1: dữ liệu

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 ký tự khác nhau. Bộ từ vựng bé xíu. Vừa trong vocab_size 4 byte. Không BPE, không rắc rối tokenizer.

### Bước 2: mô hình

Xem `code/main.py`. Block là bản sách giáo khoa từ Bài 05 — pre-norm, RMSNorm, SwiGLU, causal MHA. Số tham số cho 4/4/128: ~800K.

### Bước 3: vòng huấn luyện

Lấy ngẫu nhiên một batch các cửa sổ token dài 256. Forward. Cross-entropy dịch một vị trí. Backward. Bước AdamW. Ghi log. Lặp lại.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Bước 4: lấy mẫu

Cho một prompt, lặp đi lặp lại: forward, lấy mẫu từ logits top-p, nối vào, rồi tiếp tục. Dừng sau 500 token.

### Bước 5: đọc kết quả

Sau 2.000 bước:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Chưa phải Shakespeare. Nhưng có dáng Shakespeare. Một thắng lợi rõ ràng cho ~800K tham số và 6 phút trên laptop.

## Áp dụng

Đồ án tổng kết này là một kiến trúc tham chiếu. Ba hướng mở rộng để đưa nó ra thực tế:

1. **Thay tokenizer.** Dùng BPE (ví dụ `tiktoken.get_encoding("cl100k_base")`). Kích thước bộ từ vựng nhảy từ 65 lên ~50.000. Dung lượng mô hình phải tăng lên để bù lại.
2. **Huấn luyện trên kho ngữ liệu lớn hơn.** Dùng `OpenWebText` hoặc `fineweb-edu` (HuggingFace). 10B token trên một A100 mất khoảng 24 giờ cho GPT 125M tham số.
3. **Thêm RoPE + KV cache + Flash Attention.** Phần bài tập bên dưới dẫn bạn qua từng cái.

Kết quả cuối cùng là một GPT 125M tham số sinh tiếng Anh trôi chảy. Không phải mô hình tiên phong. Nhưng đúng đường code đó — chỉ lớn hơn — là thứ Karpathy, EleutherAI và Allen Institute dùng để huấn luyện các checkpoint (bản lưu mô hình) nghiên cứu trong năm 2026.

## Đóng gói

Xem `outputs/skill-transformer-review.md`. Skill này rà soát một bản cài đặt transformer từ đầu về tính đúng đắn dựa trên cả 13 bài trước.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Kiểm tra val loss ở bước cuối của mô hình đã huấn luyện thấp hơn 2.0. Đổi `max_steps` từ 2.000 lên 5.000 — val loss có tiếp tục giảm không?
2. **Trung bình.** Thay positional embedding học được bằng RoPE. Áp phép quay vào Q và K bên trong `MultiHeadAttention`. Huấn luyện và kiểm tra val loss ít nhất cũng thấp như trước.
3. **Trung bình.** Cài một KV cache trong vòng lấy mẫu. Sinh 500 token khi có và khi không có cache. Thời gian thực (wall-clock) sẽ cải thiện 5–20× trên laptop.
4. **Khó.** Thêm một head thứ hai dự đoán token kế-tiếp-nữa (MTP — Multi-Token Prediction từ DeepSeek-V3). Huấn luyện đồng thời. Nó có giúp ích không?
5. **Khó.** Thay FFN duy nhất mỗi block bằng MoE 4 expert. Router + định tuyến top-2. Xem val loss thay đổi thế nào khi số tham số active bằng nhau.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| nanoGPT | "repo tutorial của Karpathy" | Code huấn luyện transformer decoder-only tối giản, ~300 dòng; bản tham chiếu kinh điển. |
| tinyshakespeare | "Bộ ngữ liệu đồ chơi tiêu chuẩn" | ~1.1 MB văn bản; mọi tutorial character-LM từ 2015 tới nay đều dùng nó. |
| Tied embeddings | "Chia sẻ ma trận vào/ra" | Trọng số LM head = chuyển vị của ma trận token embedding; tiết kiệm tham số, tăng chất lượng. |
| bf16 autocast | "Mẹo độ chính xác khi huấn luyện" | Chạy forward/backward ở bf16, giữ trạng thái optimizer ở fp32; chuẩn từ 2021. |
| Gradient clipping | "Chặn các cú nhảy vọt" | Giới hạn chuẩn gradient toàn cục ở 1.0; ngăn huấn luyện nổ tung. |
| Cosine LR schedule | "Mặc định từ 2020 trở đi" | LR tăng tuyến tính (warmup) rồi giảm theo hình cosine xuống 10% đỉnh. |
| MFU | "Model FLOP Utilization" | FLOPs đạt được / đỉnh lý thuyết; 40% với mô hình dense, 30% với MoE là con số mạnh trong 2026. |
| Val loss | "Loss trên tập giữ riêng" | Cross-entropy trên dữ liệu mô hình chưa từng thấy; là máy dò overfitting. |

## Đọc thêm

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — bản cài đặt có chú thích kinh điển.

---

[← Bài trước: Quy luật tỉ lệ (Scaling Laws)](../../13-scaling-laws/docs/vi.md) · [Bài sau: Các biến thể attention →](../../15-attention-variants/docs/vi.md)
