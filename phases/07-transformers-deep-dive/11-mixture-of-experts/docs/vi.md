# Mixture of Experts (MoE)

> Một transformer dense 70B kích hoạt mọi tham số cho mỗi token. Một MoE 671B chỉ kích hoạt 37B mỗi token mà vẫn đánh bại nó trên mọi benchmark. sparse activation (kích hoạt thưa) là ý tưởng mở rộng quy mô quan trọng nhất của thập kỷ này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 05 (Transformer đầy đủ), Giai đoạn 7, Bài 07 (GPT)
**Time:** ~45 phút

## Vấn đề

FLOPs của một transformer dense khi suy luận bằng số tham số của nó (nhân 2 cho lượt forward). Phóng to một mô hình dense thì mỗi token đều phải trả hóa đơn đầy đủ. Đến 2024, biên giới đã chạm bức tường tính toán: để thông minh hơn một cách đáng kể, bạn cần FLOPs mỗi token nhiều lên theo cấp số mũ.

Mixture of Experts (hỗn hợp chuyên gia) phá vỡ mối liên kết này. Thay mỗi FFN bằng `E` expert (chuyên gia) độc lập cộng một router (bộ định tuyến) chọn `k` chuyên gia cho mỗi token. Tổng tham số = `E × FFN_size`. Tham số kích hoạt mỗi token = `k × FFN_size`. Cấu hình điển hình năm 2026: `E=256`, `k=8`. Lưu trữ tăng theo `E`, tính toán tăng theo `k`.

Biên giới năm 2026 gần như hoàn toàn là MoE: DeepSeek-V3 (671B tổng / 37B kích hoạt), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss. Trên bảng xếp hạng độc lập của Artificial Analysis, 10 mô hình mã nguồn mở hàng đầu đều là MoE.

## Khái niệm

![Lớp MoE: bộ định tuyến chọn k trong E chuyên gia cho mỗi token](../assets/moe.svg)

### Hoán đổi FFN

Khối transformer dense:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

Khối MoE:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

Mỗi chuyên gia là một FFN độc lập (thường là SwiGLU). Bộ định tuyến chỉ là một lớp tuyến tính duy nhất. Mỗi token tự chọn `k` chuyên gia của mình và nhận về một hỗn hợp có cổng từ đầu ra của chúng.

### Vấn đề cân bằng tải

Nếu bộ định tuyến đẩy 90% token qua chuyên gia 3, các chuyên gia còn lại sẽ chết đói. Ba cách sửa đã được thử:

1. **Mất mát phụ trợ load balancing (cân bằng tải)** (Switch Transformer, Mixtral). Thêm một hình phạt tỉ lệ với phương sai trong mức dùng của các chuyên gia. Hiệu quả, nhưng thêm một siêu tham số và một tín hiệu gradient thứ hai.
2. **Sức chứa chuyên gia + bỏ token** (Switch đời đầu). Mỗi chuyên gia xử lý tối đa `C × N/E` token; token tràn sẽ bỏ qua lớp đó. Làm giảm chất lượng.
3. **Cân bằng không cần mất mát phụ trợ** (DeepSeek-V3). Thêm một độ lệch (bias) học được cho từng chuyên gia, dịch chuyển lựa chọn top-k của bộ định tuyến. Bias được cập nhật ngoài mất mát huấn luyện. Không có hình phạt nào lên mục tiêu chính. Bước đột phá lớn của năm 2024.

Cách làm của DeepSeek-V3: sau mỗi bước huấn luyện, với từng chuyên gia, kiểm tra xem mức dùng của nó cao hơn hay thấp hơn mục tiêu. Nhích bias thêm `±γ`. Lựa chọn dùng `scores + bias`. Xác suất chuyên gia dùng cho top-k gating (cổng chọn top-k) vẫn là `scores` thô, không đổi. Định tuyến tách rời khỏi biểu đạt.

### Chuyên gia chia sẻ

DeepSeek-V2/V3 còn chia chuyên gia thành *chia sẻ* và *được định tuyến*. Mọi token đều đi qua tất cả chuyên gia chia sẻ. Chuyên gia được định tuyến thì chọn qua top-k. Chuyên gia chia sẻ nắm kiến thức phổ biến; chuyên gia được định tuyến thì chuyên môn hóa. V3 chạy 1 chuyên gia chia sẻ cộng top-8 trong 256 chuyên gia được định tuyến.

### Chuyên gia hạt mịn

MoE cổ điển (GShard, Switch): mỗi chuyên gia rộng bằng một FFN đầy đủ. `E` nhỏ (8–64), `k` nhỏ (1–2).

MoE hạt mịn hiện đại (DeepSeek-V3, Qwen-MoE): mỗi chuyên gia hẹp hơn (1/8 kích thước FFN). `E` lớn (256+), `k` lớn hơn (8+). Cùng tổng tham số, nhưng số tổ hợp tăng nhanh hơn nhiều. `C(256, 8) = 400 trillion` "chuyên gia" khả dĩ cho mỗi token. Chất lượng tăng, độ trễ giữ nguyên.

### Hồ sơ chi phí

Mỗi token, mỗi lớp:

| Cấu hình | Tham số kích hoạt / token | Tổng tham số |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 đánh bại Llama 3 70B (dense) trên gần như mọi benchmark trong khi **tốn ít FLOPs kích hoạt hơn trên mỗi token**. Nhiều tham số hơn = nhiều kiến thức hơn. Nhiều FLOPs kích hoạt hơn = nhiều tính toán hơn mỗi token. MoE tách rời hai điều đó.

### Cái bẫy: bộ nhớ

Mọi chuyên gia đều nằm trên GPU bất kể chuyên gia nào thực sự chạy. Một mô hình 671B cần ~1.3 TB VRAM cho trọng số fp16. Triển khai MoE ở biên giới cần song song chuyên gia — chia nhỏ chuyên gia ra nhiều GPU, định tuyến token qua mạng. Độ trễ bị chi phối bởi giao tiếp all-to-all, không phải phép nhân ma trận.

```figure
expert-routing
```

## Tự tay làm

Xem `code/main.py`. Một lớp MoE gọn nhẹ bằng stdlib thuần với:

- `n_experts=8` chuyên gia kiểu SwiGLU (mỗi cái một lớp tuyến tính, để minh họa)
- định tuyến top-k=2
- trọng số cổng chuẩn hóa bằng softmax
- cân bằng không cần mất mát phụ trợ nhờ bias riêng cho từng chuyên gia

### Bước 1: bộ định tuyến

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias ảnh hưởng đến lựa chọn, không ảnh hưởng đến trọng số cổng. Đó là mẹo của DeepSeek-V3 — bias chỉnh mất cân bằng tải mà không lái dự đoán của mô hình.

### Bước 2: đưa 100 token qua bộ định tuyến

Theo dõi chuyên gia nào chạy bao nhiêu lần. Không có bias, mức dùng bị lệch. Với vòng lặp cập nhật bias (`-γ` cho chuyên gia bị dùng quá nhiều, `+γ` cho chuyên gia bị dùng quá ít), mức dùng hội tụ về phân phối đều sau vài vòng lặp.

### Bước 3: so sánh số tham số

In ra "tương đương dense" của một cấu hình MoE. Dạng DeepSeek-V3: 256 được định tuyến + 1 chia sẻ, 8 kích hoạt, d_model=7168. Tổng số tham số thì chóng mặt. Số tham số kích hoạt chỉ bằng một phần bảy của Llama 3 70B dense.

## Áp dụng

Nạp bằng HuggingFace:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

Suy luận production năm 2026: vLLM hỗ trợ định tuyến MoE sẵn trong lõi. SGLang có đường song song chuyên gia nhanh nhất. Cả hai tự xử lý lựa chọn top-k và song song chuyên gia.

**Khi nào nên chọn MoE:**
- Bạn muốn chất lượng ở biên giới với chi phí suy luận mỗi token thấp hơn.
- Bạn có VRAM / hạ tầng song song chuyên gia.
- Khối lượng công việc của bạn nặng về token (chat, code) chứ không nặng về ngữ cảnh (tài liệu dài).

**Khi nào KHÔNG nên chọn MoE:**
- Triển khai trên thiết bị biên — bạn trả toàn bộ chi phí lưu trữ cho từng FLOP kích hoạt.
- Phục vụ một người dùng với yêu cầu độ trễ khắt khe — định tuyến chuyên gia thêm chi phí.
- Mô hình nhỏ (<7B) — lợi thế chất lượng của MoE chỉ xuất hiện khi vượt một ngưỡng tính toán (~6B tham số kích hoạt).

## Đóng gói

Xem `outputs/skill-moe-configurator.md`. Skill chọn E, k, và bố cục chuyên gia chia sẻ cho một MoE mới, dựa trên ngân sách tham số, số token huấn luyện và mục tiêu triển khai.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Xem cách cập nhật bias không cần mất mát phụ trợ làm đều mức dùng chuyên gia qua 50 vòng lặp.
2. **Trung bình.** Thay bộ định tuyến học được bằng bộ định tuyến dựa trên hàm băm (tất định, không học). So sánh chất lượng và độ cân bằng. Vì sao bộ định tuyến học được lại tốt hơn?
3. **Khó.** Cài đặt "định tuyến khớp rollout" kiểu GRPO (mẹo của DeepSeek-V3.2): ghi lại chuyên gia nào chạy khi suy luận, ép định tuyến giống hệt khi tính gradient. Đo tác động trên một thiết lập policy-gradient đồ chơi.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Expert | "Một FFN trong nhiều FFN" | Một mạng feed-forward độc lập; tham số dành riêng cho một lát thưa của phép tính FFN. |
| Router | "Cái cổng" | Một lớp tuyến tính tí hon chấm điểm từng token với từng chuyên gia; lựa chọn top-k. |
| Top-k routing | "k chuyên gia kích hoạt mỗi token" | Phép tính FFN của mỗi token đi qua đúng k chuyên gia, có trọng số theo cổng. |
| Auxiliary loss | "Hình phạt cân bằng tải" | Số hạng mất mát thêm vào, phạt mức dùng chuyên gia bị lệch. |
| Auxiliary-loss-free | "Mẹo của DeepSeek-V3" | Cân bằng nhờ bias riêng cho từng chuyên gia, chỉ trên phần lựa chọn của bộ định tuyến; không thêm gradient. |
| Shared expert | "Luôn bật" | Chuyên gia thêm mà mọi token đều đi qua; nắm kiến thức phổ biến. |
| Expert parallelism | "Chia theo chuyên gia" | Phân tán các chuyên gia khác nhau lên các GPU khác nhau; định tuyến token qua mạng. |
| Sparsity | "Tham số kích hoạt < tổng tham số" | Tỉ lệ `k × expert_size / (E × expert_size)`; 37/671 ≈ 5.5% với DeepSeek-V3. |

## Đọc thêm

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — ý tưởng khởi nguồn.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch, MoE cổ điển.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + MoE không cần mất mát phụ trợ + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — bài báo về cân bằng dựa trên bias.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — cách tách chuyên gia hạt mịn + chia sẻ mà bộ định tuyến của bài này dùng.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — bài báo gốc về chuyên gia chia sẻ.

---

[← Bài trước: Audio Transformer (transformer cho âm thanh)](../../10-audio-transformers-whisper/docs/vi.md) · [Bài sau: KV Cache, Flash Attention & Tối ưu suy luận →](../../12-kv-cache-flash-attention/docs/vi.md)
