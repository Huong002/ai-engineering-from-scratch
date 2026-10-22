# Quy luật tỉ lệ (Scaling Laws)

> Bài báo Kaplan 2020 nói: mô hình lớn hơn, loss thấp hơn. Bài báo Hoffmann 2022 nói: bạn đã huấn luyện thiếu. Compute đổ vào hai chỗ — tham số và token — và cách chia không hề hiển nhiên.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 05 (Transformer đầy đủ), Giai đoạn 7, Bài 07 (GPT)
**Time:** ~45 phút

## Vấn đề

Khi bạn có C FLOPs ngân sách tính toán (compute budget) để huấn luyện và muốn có mô hình tốt nhất, bạn đứng trước hai núm vặn:

1. **Bao nhiêu tham số (N)?** Mô hình lớn hơn, dung lượng cao hơn.
2. **Bao nhiêu token huấn luyện (D)?** Nhiều dữ liệu hơn, dùng dung lượng tốt hơn.

FLOPs tỉ lệ xấp xỉ `6 × N × D`. Bạn có thể đẩy N lên và D xuống, hoặc D lên và N xuống. Cách nào tốt hơn?

Trước 2022, câu trả lời là "đẩy N thật mạnh." GPT-3 (2020) có 175B tham số huấn luyện trên ~300B token. Tỉ lệ khoảng 1,7 token trên mỗi tham số (tokens per parameter). Các quy luật tỉ lệ của Kaplan củng cố điều này.

Hoffmann et al. (2022), khi huấn luyện một họ mô hình nhỏ tên là Chinchilla, tìm ra điều khác: tỉ lệ tối ưu gần với **20 token trên mỗi tham số**. GPT-3 bị huấn luyện thiếu 10×. Chinchilla (70B tham số, 1,4T token) đánh bại GPT-3 (175B, 300B token) trên mọi benchmark với chi phí suy luận thấp hơn 2,5×.

Năm 2026 là thế giới của Chinchilla — với một khúc quanh quan trọng. Llama 3 8B được huấn luyện trên 15 nghìn tỉ token, tỉ lệ 1.875 token trên mỗi tham số. Gấp chín mươi bốn lần mức tối ưu Chinchilla. Với những mô hình sẽ được dùng ở quy mô lớn, chi phí suy luận quan trọng hơn chi phí huấn luyện, nên huấn luyện quá mức (vượt qua Chinchilla) để có bộ triển khai nhỏ hơn là mặc định của năm 2026.

## Khái niệm

![Đường cong Chinchilla: loss so với compute ở nhiều tỉ lệ N/D khác nhau](../assets/scaling-laws.svg)

### Quy luật Hoffmann

Từ bài báo Chinchilla, loss tuân theo:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = tham số (không tính embedding).
- `D` = token huấn luyện.
- `α ≈ 0.34`, `β ≈ 0.28` (gần như đối xứng).
- `E ≈ 1.69`, ngưỡng loss không thể giảm thêm.
- `A ≈ 406`, `B ≈ 411`.

Hai số hạng đánh đổi lẫn nhau khi bạn tăng quy mô. Lấy đạo hàm theo `N` với compute cố định (C = 6ND) và giải:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Tối ưu theo chi phí tính toán (compute-optimal): 20 token trên mỗi tham số.

### Vì sao vẫn huấn luyện quá mức

Chinchilla-optimal tối thiểu hoá loss huấn luyện trên mỗi FLOP huấn luyện. Nhưng bạn trả chi phí huấn luyện một lần; chi phí suy luận thì mãi mãi.

Với một chatbot phục vụ một nghìn tỉ token mỗi tháng, suy luận chiếm phần lớn tổng chi phí. Cách của Llama: huấn luyện mô hình nhỏ hơn, lâu hơn. 8B với 15T token là cực kỳ tối ưu cho suy luận:

- Vừa trên GPU tiêu dùng.
- Độ trễ chỉ bằng một phần nhỏ của Chinchilla-optimal 70B.
- Chất lượng đủ gần cho hầu hết tác vụ.

Bài báo 2024 của DeepMind ("Over-training is the new optimal") chính thức hoá điều này. Với khối lượng công việc mà suy luận chiếm ưu thế, tỉ lệ đúng gần với 100–500 token trên mỗi tham số tuỳ theo sản lượng phục vụ.

### Đột sinh so với mượt mà

Có ý kiến cho rằng: một số năng lực (số học, suy luận nhiều bước, làm theo chain-of-thought) "đột sinh" (emergent ability) một cách đột ngột ở một quy mô nào đó.

Schaeffer et al. (2023) lập luận đây là hiện tượng của cách đo: các chỉ số đột sinh dùng cách cho điểm gián đoạn (khớp chính xác, độ chính xác theo ngưỡng), che mất sự cải thiện mượt mà trong logits bên dưới. Các chỉ số liên tục (cross-entropy) cho thấy đường cong mượt.

Năm 2026, đồng thuận là: dự đoán qua loss liên tục thì đáng tin. Các bước nhảy trên benchmark thường là hiện tượng của bộ chấm điểm. Hãy lập ngân sách dựa trên các chỉ số liên tục.

### Bức tranh năm 2026

Quy luật tỉ lệ vẫn đúng, nhưng:

| Yếu tố | Thay đổi thế nào |
|--------|-------------|
| Chất lượng dữ liệu | Lọc các token "tốt" (kiểu Phi) dịch chuyển đường cong tương đương hơn 2× compute hiệu dụng |
| MoE | Tổng tham số tách khỏi FLOPs hoạt động; quy luật tỉ lệ tính theo FLOP hoạt động |
| Post-training | Một số năng lực (làm theo chỉ dẫn, viết code) dịch chuyển nhờ SFT+RLHF nhiều hơn nhờ pretraining |
| Đa phương thức | Token ảnh + văn bản tăng cùng nhau; đường cong riêng cho từng phương thức |
| Dữ liệu tổng hợp | Mô hình sinh dữ liệu huấn luyện; compute hiệu dụng có thể nhân lên |

Optimizer Muon (Kimi Moonlight, 2024) cho thấy mức tăng compute hiệu dụng ~2× so với AdamW với cùng lượng dữ liệu. Một số lần chạy huấn luyện năm 2026 dùng Muon mặc định. Nó thay đổi hằng số tuyệt đối trong quy luật tỉ lệ, không thay đổi hình dạng.

```figure
scaling-laws
```

## Tự tay làm

Xem `code/main.py`. Ta cài đặt phương trình loss của Chinchilla và giải tìm `(N, D)` tối ưu theo chi phí tính toán ở từng mức ngân sách tính toán.

### Bước 1: Loss của Chinchilla

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

Vẽ `L` thành đường đồng mức trên `(N, D)` với `C = 6ND` cố định. Tìm cực tiểu.

### Bước 2: biên tối ưu theo chi phí tính toán

Với ngân sách tính toán từ `1e17` đến `1e25` FLOPs, tìm `(N, D)` tối thiểu hoá loss với ràng buộc `6ND = C`. Kiểm chứng tỉ lệ `D/N ≈ 20`.

### Bước 3: chi phí huấn luyện quá mức

Tính phần loss tăng thêm khi huấn luyện một mô hình nhỏ hơn 10× (N bằng 1/10 mức tối ưu, D gấp 10× mức tối ưu). Báo cáo khoản tiết kiệm FLOPs khi suy luận (tỉ lệ với N) để đổi lại.

### Bước 4: so sánh với các mô hình thật

Đưa vào các cặp `(N, D)` đã biết của GPT-3, Chinchilla, Llama 3 8B, DeepSeek-V3 (tham số hoạt động), rồi so sánh loss dự đoán với loss được báo cáo.

## Áp dụng

Bạn khó có khả năng tự huấn luyện một mô hình tiên phong. Nhưng quy luật tỉ lệ cho bạn biết:

1. **Bản fine-tune của bạn có đủ dữ liệu chưa.** Nếu dữ liệu riêng cho tác vụ thấp hơn 20 token trên mỗi tham số của mô hình gốc, hãy dự kiến bão hoà ở một ngưỡng loss nào đó.
2. **Có nên chọn mô hình gốc lớn hơn.** Nếu bạn tiêu hết ngân sách cho suy luận, hãy chọn mô hình nhỏ hơn, huấn luyện lâu hơn.
3. **Chỗ nào lợi suất giảm dần.** Vượt quá 1000× mức tối ưu Chinchilla, các thay đổi log-loss trở thành nhiễu.

**Quỹ đạo nghiên cứu năm 2026:**

- **Chế độ bị dữ liệu giới hạn.** Web có số token chất lượng cao hữu hạn (~5–10 nghìn tỉ tiếng Anh sau khi lọc). Pretraining tiên phong đang tiến tới trần này. Dữ liệu tổng hợp, đa ngôn ngữ, đa phương thức, và fine-tuning mở rộng bằng RLHF là các đòn tiếp theo.
- **Chiêu nhân compute.** Optimizer Muon, MoE, lọc dữ liệu tốt hơn — mỗi thứ dịch chuyển hằng số tuyệt đối, không dịch chuyển tiệm cận.
- **Quy luật tỉ lệ cho RL.** Câu hỏi mở. Bằng chứng ban đầu cho thấy quy luật luỹ thừa (power law) trên số mẫu RL nhưng với số mũ rất khác so với pretraining.

## Đóng gói

Xem `outputs/skill-training-budget-estimator.md`. Skill này chọn `(N, D, hours, GPU)` cho một lần huấn luyện mới dựa trên ngân sách tính toán, ràng buộc triển khai, và loss mục tiêu.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. In `(N, D)` tối ưu Chinchilla cho các ngân sách tính toán `1e20`, `1e22`, `1e24`. So sánh với bảng mô hình thật.
2. **Trung bình.** Cài đặt đường cong loss-theo-compute của Hoffmann. Vẽ loss so với `log10(C)` cho biên tối ưu theo chi phí tính toán. Xác định khi nào quy luật dự đoán ta cần `>10^28` FLOPs để giảm thêm 0.1 cross-entropy.
3. **Khó.** Tự khớp một quy luật tỉ lệ trên 5 mô hình tí hon (100K đến 10M tham số) huấn luyện trên cùng bộ dữ liệu. Ước lượng `α` và `E`. Số mũ của bạn khớp với số mũ công bố đến mức nào?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Parameters (N) | "Kích thước mô hình" | Số trọng số không tính embedding; quyết định dung lượng. |
| Tokens (D) | "Dữ liệu huấn luyện" | Số token huấn luyện đã thấy; quyết định tham số được dùng tốt đến đâu. |
| Compute (C) | "FLOPs đã tiêu" | Xấp xỉ `6 × N × D` với một transformer chuẩn. |
| Chinchilla-optimal | "D/N ≈ 20" | Tỉ lệ tối thiểu hoá loss trên mỗi FLOP pretraining. |
| Over-training | "Vượt qua Chinchilla" | Tiêu thêm FLOPs huấn luyện để tiết kiệm FLOPs suy luận; D/N >> 20. |
| Irreducible loss | "Cái sàn" | Số hạng `E` trong quy luật tỉ lệ; chính là entropy của dữ liệu. |
| Emergent capability | "Nhảy đột ngột khi tăng quy mô" | Thường là hiện tượng của bộ chấm điểm; loss liên tục thì mượt. |
| Effective compute | "Hệ số nhân hiệu quả huấn luyện" | Dữ liệu / optimizer / kiến trúc tốt hơn nhân lên quãng đường một FLOP đi được. |

## Đọc thêm

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — bài báo quy luật tỉ lệ đầu tiên; huấn luyện thiếu.
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla.
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — đột sinh như một hiện tượng của phép đo.
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — vì sao việc huấn luyện quá mức của Llama là đúng cho khối lượng công việc của nó.
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — hệ số nhân compute 2×.

---

[← Bài trước: KV Cache, Flash Attention & Tối ưu suy luận](../../12-kv-cache-flash-attention/docs/vi.md) · [Bài sau: Tự xây dựng Transformer từ đầu →](../../14-build-a-transformer-capstone/docs/vi.md)
