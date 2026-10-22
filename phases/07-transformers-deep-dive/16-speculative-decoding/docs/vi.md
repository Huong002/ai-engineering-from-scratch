# Speculative decoding (giải mã suy đoán) — Nháp, xác minh, lặp lại

> Giải mã tự hồi quy là tuần tự. Mỗi token phải chờ token trước nó. Giải mã suy đoán phá vỡ chuỗi đó: một mô hình nháp (draft model) rẻ tiền đề xuất N token, mô hình đắt tiền xác minh (verification) cả N token trong một lượt forward. Khi bản nháp đúng, bạn chỉ trả một lượt forward lớn cho N token được sinh ra.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 07 (GPT causal LM), Giai đoạn 7, Bài 12 (KV cache & Flash attention)
**Time:** ~60 phút

## Vấn đề

Một LLM 70B lấy mẫu một token mất ~30 ms trên H100. Một mô hình nháp 3B mất ~3 ms. Nếu để mô hình 3B nháp trước 5 token, rồi chạy mô hình 70B *một lần* để xác minh cả 5, tổng là `5×3 + 30 = 45 ms` cho tối đa 5 token được chấp nhận — so với `5×30 = 150 ms` khi sinh tuần tự. Đó chính là toàn bộ lời chào của giải mã suy đoán: đổi một chút bộ nhớ GPU thêm (mô hình nháp) lấy độ trễ (latency) giải mã thấp hơn 2–4 lần.

Mẹo này phải giữ nguyên phân phối. Speculative sampling, do Leviathan et al. (2023) và Chen et al. công bố đồng thời, bảo đảm chuỗi đầu ra có **phân phối y hệt** như khi mô hình lớn tự sinh một mình — tức là không mất mát phân phối (lossless). Không đánh đổi chất lượng. Chỉ nhanh hơn.

Bốn nhóm cặp nháp–xác minh thống trị suy luận năm 2026:

1. **Speculative vanilla (Leviathan 2023).** Một mô hình nháp riêng (ví dụ Llama 3 1B) + một mô hình xác minh (ví dụ Llama 3 70B).
2. **Medusa (Cai 2024).** Nhiều head giải mã trên mô hình xác minh dự đoán song song các vị trí `t+1..t+k`. Không cần mô hình nháp riêng.
3. **Họ EAGLE (Li 2024, 2025).** Bản nháp nhẹ tái dùng hidden state của mô hình xác minh; tỉ lệ chấp nhận (acceptance rate) cao hơn vanilla; thường đạt 3–4×.
4. **Lookahead decoding (Fu 2024).** Lặp Jacobi; hoàn toàn không cần mô hình nháp. Tự suy đoán. Ít phổ biến nhưng không phụ thuộc gì.

Mọi stack suy luận production năm 2026 mặc định bật giải mã suy đoán. vLLM, TensorRT-LLM, SGLang và llama.cpp đều hỗ trợ ít nhất vanilla + EAGLE-2.

## Khái niệm

### Thuật toán cốt lõi

Cho một mô hình xác minh `M_q` và một mô hình nháp rẻ hơn `M_p`:

1. Gọi `x_1..x_k` là tiền tố đã giải mã xong.
2. **Nháp**: dùng `M_p` tự hồi quy đề xuất `d_{k+1}, d_{k+2}, ..., d_{k+N}` với xác suất nháp `p_1..p_N`.
3. **Xác minh song song**: chạy `M_q` một lần trên `x_1..x_k, d_{k+1}, ..., d_{k+N}`, thu được xác suất của mô hình xác minh `q_1..q_{N+1}` cho các vị trí `k+1..k+N+1`.
4. **Chấp nhận/từ chối từng token nháp từ trái sang phải**: với mỗi `i`, chấp nhận với xác suất `min(1, q_i(d_i) / p_i(d_i))`.
5. Tại lần từ chối đầu tiên ở vị trí `j`: lấy mẫu `t_j` từ phân phối phần dư (residual distribution) `(q_j - p_j)_+` đã chuẩn hoá. Mọi bản nháp sau `j` bị bỏ.
6. Nếu chấp nhận cả `N`: lấy mẫu thêm một token `t_{N+1}` từ `q_{N+1}` (token thưởng miễn phí).

Phân phối phần dư chính là điểm mấu chốt toán học giúp đầu ra vẫn có phân phối đúng hệt như khi `M_q` tự lấy mẫu từ đầu.

### Điều gì quyết định tốc độ tăng

Gọi `α` = tỉ lệ chấp nhận kỳ vọng trên mỗi token nháp. Gọi `c` = tỉ lệ chi phí nháp so với xác minh. Mỗi bước:

- Sinh ngây thơ gọi mô hình lớn 1 lần cho mỗi token.
- Giải mã suy đoán gọi mô hình lớn 1 lần cho mỗi `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` token khi `α` cao.

Quy tắc ngón tay cái ở `α = 0.75` và `N = 5`: số lần gọi mô hình lớn giảm 3×. Chi phí nháp rẻ 5×. Thời gian thực (wall-clock) tổng cộng giảm ~2.5×.

**α phụ thuộc vào:**

- Bản nháp xấp xỉ mô hình xác minh tốt đến đâu. Cùng họ / cùng dữ liệu huấn luyện giúp α tăng rõ rệt.
- Chiến lược giải mã. Nháp greedy với xác minh greedy: α cao. Lấy mẫu theo temperature: khó khớp hơn; tỉ lệ chấp nhận giảm.
- Loại tác vụ. Code và đầu ra có cấu trúc được chấp nhận nhiều hơn (dễ đoán); viết sáng tạo tự do được chấp nhận ít hơn.

### Medusa — nháp mà không cần mô hình nháp

Medusa thay mô hình nháp bằng các head đầu ra phụ trên mô hình xác minh. Tại vị trí `t`:

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

Mỗi head xuất ra logits riêng. Khi suy luận, bạn lấy mẫu từ mỗi head để có một chuỗi ứng viên, rồi xác minh bằng một lượt forward dùng sơ đồ tree attention xem xét tất cả phần tiếp nối ứng viên cùng lúc.

Ưu điểm: không cần mô hình thứ hai. Nhược điểm: thêm tham số huấn luyện được; cần một giai đoạn fine-tuning có giám sát (~1B token); tỉ lệ chấp nhận thấp hơn một chút so với speculative vanilla có bản nháp tốt.

### EAGLE — bản nháp tốt hơn nhờ tái dùng hidden state

EAGLE-1/2/3 (Li et al., 2024–2025) biến mô hình nháp thành một transformer tí hon (thường 1 lớp) nhận hidden state lớp cuối của mô hình xác minh. Vì bản nháp thấy được biểu diễn đặc trưng của mô hình xác minh, dự đoán của nó tương quan mạnh với phân phối đầu ra của mô hình xác minh. Tỉ lệ chấp nhận tăng từ ~0.6 (vanilla) lên 0.85+.

EAGLE-3 (2025) thêm tìm kiếm dạng cây trên các phần tiếp nối ứng viên. vLLM và SGLang đưa EAGLE-2/3 thành đường suy đoán mặc định cho Llama 3/4 và Qwen 3.

### Vũ điệu KV cache

Bước xác minh nhét `N` token nháp vào mô hình xác minh trong một lượt forward. Việc này mở rộng KV cache của mô hình xác minh thêm `N` mục. Nếu vài bản nháp bị từ chối, bạn phải lùi cache về đúng độ dài tiền tố được chấp nhận.

Các triển khai production (`--speculative-config` của vLLM, LookaheadDecoder của TensorRT-LLM) xử lý việc này bằng buffer KV nháp. Ghi trước, chốt khi được chấp nhận. Không khó về mặt khái niệm, nhưng lắt nhắt.

```figure
draft-verify-tokens
```

## Tự tay làm

Xem `code/main.py`. Chúng ta hiện thực thuật toán speculative sampling cốt lõi (bước từ chối + phân phối phần dư) với:

- Một "mô hình lớn" là softmax tất định trên một phân phối viết tay (để kiểm tra phần chấp nhận bằng giải tích).
- Một "mô hình nháp" là một nhiễu loạn của mô hình lớn.
- Một vòng lặp chấp nhận / từ chối cho ra đúng phân phối biên như khi lấy mẫu trực tiếp.

### Bước 1: bước từ chối

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` là một số ngẫu nhiên đều. `q_prob` là xác suất của mô hình xác minh cho token được nháp. `p_prob` là xác suất của mô hình nháp. Định lý Leviathan nói rằng quyết định Bernoulli này, theo sau là lấy mẫu từ phân phối phần dư khi bị từ chối, giữ nguyên chính xác phân phối của mô hình xác minh.

### Bước 2: phân phối phần dư

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

Trừ `p` khỏi `q` theo từng phần tử, kẹp giá trị âm về 0, chuẩn hoá lại. Lấy mẫu từ phân phối này mỗi khi có từ chối.

### Bước 3: một bước suy đoán

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

Năm token được chấp nhận → một token thưởng → sáu token sinh ra trong một lượt của mô hình xác minh.

### Bước 4: đo tỉ lệ chấp nhận

Chạy 10.000 bước suy đoán với các mức chất lượng nháp khác nhau. Vẽ tỉ lệ chấp nhận theo KL divergence giữa phân phối nháp và phân phối xác minh. Bạn sẽ thấy một quan hệ đơn điệu rõ ràng.

### Bước 5: kiểm chứng tương đương phân phối

Theo thực nghiệm: histogram token do vòng lặp suy đoán sinh ra phải khớp với histogram khi lấy mẫu trực tiếp từ mô hình xác minh. Đó là định lý Leviathan trong thực tế. Kiểm định chi-square xác nhận trong sai số lấy mẫu.

## Áp dụng

Production:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "eagle", "model": "/models/llama-3.1-eagle-70b", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 5}'

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "draft_model", "model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 5}'
```

TensorRT-LLM có đường Medusa nhanh nhất tính đến giữa 2026. `faster-whisper` bọc giải mã suy đoán cho Whisper-large với một bản nháp nhỏ.

**Chọn bản nháp:**

| Chiến lược | Khi nào nên chọn | Tốc độ tăng |
|----------|--------------|---------|
| Nháp vanilla (họ Llama 1B/3B) | Prototype nhanh, không cần huấn luyện | 1.8–2.3× |
| Head Medusa | Bạn fine-tune được mô hình xác minh | 2–3× |
| EAGLE-2 / 3 | Production, tốc độ tối đa | 3–4× |
| Lookahead | Không nháp, không huấn luyện, không thêm tham số | 1.3–1.6× |

**Khi nào KHÔNG nên dùng giải mã suy đoán:**

- Sinh một chuỗi chỉ 1–5 token. Overhead chiếm ưu thế.
- Lấy mẫu sáng tạo / temperature cao (α giảm).
- Triển khai bị hạn chế bộ nhớ (mô hình nháp thêm VRAM).

## Đóng gói

Xem `outputs/skill-spec-decode-picker.md`. Skill này chọn một chiến lược giải mã suy đoán (vanilla / Medusa / EAGLE / lookahead) cùng các tham số tinh chỉnh (N, temperature của bản nháp) cho một khối lượng suy luận mới.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Xác nhận phân phối token suy đoán khớp với phân phối lấy mẫu trực tiếp của mô hình xác minh trên 50.000 token, với chi-square p > 0.05.
2. **Trung bình.** Vẽ tốc độ tăng (số token trên mỗi lượt forward của mô hình lớn) theo `N` với `α = 0.5, 0.7, 0.85`. Xác định `N` tối ưu cho mỗi α. (Gợi ý: số token kỳ vọng mỗi lần xác minh = `(1 - α^{N+1}) / (1 - α)`.)
3. **Khó.** Hiện thực một Medusa tí hon: lấy GPT capstone từ Bài 14, thêm 3 head LM phụ dự đoán các vị trí t+2, t+3, t+4. Huấn luyện trên tinyshakespeare với một loss đa head chung. So sánh tỉ lệ chấp nhận với một bản nháp vanilla tạo bằng cách cắt bớt chính mô hình đó.
4. **Khó.** Hiện thực rollback: bắt đầu với KV cache tiền tố 10 token, nạp 5 token nháp, mô phỏng bị từ chối ở vị trí 3. Kiểm chứng rằng các lần đọc cache của bạn khớp đúng "tiền tố + 2 bản nháp đầu được chấp nhận" ở vòng lặp kế tiếp.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Draft model | "Cái rẻ tiền" | Một mô hình nhỏ hơn đề xuất token ứng viên; thường rẻ hơn mô hình xác minh 10–50×. |
| Verifier | "Cái to" | Mô hình đích mà ta giữ nguyên phân phối; chạy một lần mỗi bước suy đoán. |
| Acceptance rate (α) | "Bản nháp đúng bao nhiêu lần" | Xác suất trên mỗi token mà mô hình xác minh chấp nhận bản nháp. Thường 0.7–0.9. |
| Residual distribution | "Đường lui khi bị từ chối" | `(q - p)_+` đã chuẩn hoá; lấy mẫu từ đó khi bị từ chối sẽ giữ nguyên phân phối của mô hình xác minh. |
| Bonus token | "Cái miễn phí" | Khi cả N bản nháp được chấp nhận, lấy mẫu thêm một token từ phân phối bước kế của mô hình xác minh. |
| Medusa | "Suy đoán không cần bản nháp" | Nhiều head LM trên mô hình xác minh dự đoán song song các vị trí t+1..t+k. |
| EAGLE | "Bản nháp từ hidden state" | Transformer nháp tí hon, điều kiện hoá trên hidden state lớp cuối của mô hình xác minh. |
| Lookahead decoding | "Lặp Jacobi" | Tự suy đoán bằng một phép lặp điểm bất động; không cần mô hình nháp. |
| Tree attention | "Xác minh nhiều ứng viên cùng lúc" | Xác minh dạng phân nhánh, xem xét nhiều phần tiếp nối nháp cùng lúc. |
| KV rollback | "Hoàn tác các bản nháp bị từ chối" | Buffer KV nháp; chốt khi được chấp nhận, bỏ khi bị từ chối. |

## Đọc thêm

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — thuật toán cốt lõi và định lý tương đương.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — công bố đồng thời; chứng minh từ chối Bernoulli gọn gàng.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — bài báo Medusa; xác minh bằng tree attention.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1; bản nháp điều kiện hoá trên hidden state.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2; độ sâu cây động.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead, cách tiếp cận không cần bản nháp.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — tài liệu tham chiếu production chuẩn, với cả bốn chiến lược được nối sẵn.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — code tham chiếu cho EAGLE-1/2/3.

---

[← Bài trước: Các biến thể attention](../../15-attention-variants/docs/vi.md)
