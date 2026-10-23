# Giải mã suy đoán và EAGLE-3

> Giai đoạn 7 · Bài 16 đã chứng minh phần toán: quy tắc loại bỏ Leviathan bảo toàn chính xác phân phối của bộ xác minh. Bài này là góc nhìn từ tầng huấn luyện của speculative decoding (giải mã suy đoán) trong sản xuất năm 2026. EAGLE-3 đã biến draft model (mô hình nháp) từ một xấp xỉ rẻ tiền thành một mạng nhỏ chuyên dụng, huấn luyện trên chính hidden state của bộ xác minh, rồi thêm một vòng kiểm tra lúc huấn luyện để căn khớp phân phối huấn luyện và phân phối suy luận. Kết quả: speedup (hệ số tăng tốc) đầu-cuối từ 3× đến 6.5×, tỉ lệ token được chấp nhận trên 0.9 với chat, không đánh đổi phân phối. Mọi stack suy luận production năm 2026 đều mặc định dùng nó.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 7, Bài 16 (toán giải mã suy đoán), Giai đoạn 10, Bài 12 (tối ưu suy luận)
**Time:** ~75 phút

## Mục tiêu học tập

- Phát biểu định lý Leviathan trong một câu và chứng minh rằng vòng lặp suy đoán sinh ra các mẫu có phân phối giống hệt phân phối của bộ xác minh.
- Đi qua chặng tiến hoá hai năm từ spec-decoding nguyên bản (Leviathan 2023) qua EAGLE, EAGLE-2 và EAGLE-3, gọi tên đúng hạn chế mà mỗi bước đã loại bỏ.
- Tính hệ số tăng tốc kỳ vọng từ acceptance rate (tỉ lệ chấp nhận) `α` và tỉ lệ chi phí nháp-trên-xác minh `c`, rồi chọn độ dài nháp tối ưu `N` cho từng chế độ.
- Cài đặt toàn bộ vòng lặp suy đoán từ đầu: nháp, xác minh, lấy mẫu loại bỏ từ phần dư, cuộn KV cache lại khi bị loại, phát token thưởng khi được chấp nhận toàn bộ.

## Vấn đề

Giải mã tự hồi quy trên một mô hình 70B chạy được chừng 35 token mỗi giây trên H100. GPU còn xa mới bão hoà. Băng thông bộ nhớ là trần: mỗi token nạp 70B trọng số từ HBM, làm một bước số học, và sinh ra một số thực. Các đơn vị tính toán phần lớn ngồi không.

Giải mã suy đoán biến điều đó thành một bài toán thông lượng có thể giải được. Một mô hình nháp rẻ tiền đề xuất `N` token trong `N` lượt forward nhỏ. Bộ xác minh chạy một lần trên tiền tố cộng toàn bộ `N` token nháp. Nếu phân phối của bộ xác minh tại vị trí `i` khớp với mô hình nháp (theo nghĩa thống kê mà ta sẽ nói rõ), ta chấp nhận; nếu không, ta loại và lấy mẫu một bản sửa từ phân phối phần dư. Một lượt forward lớn của mô hình lớn sinh ra tới `N+1` token được chấp nhận thay vì một token.

Định lý quan trọng là của Leviathan, Kalman, Matias (ICML 2023): phân phối đầu ra giống hệt phân phối mà việc lấy mẫu trực tiếp từ bộ xác minh sẽ tạo ra. Không phải xấp xỉ. Mà giống hệt. Đây chính là lý do giải mã suy đoán được chấp nhận trong production — nó là tối ưu thuần tuý về độ trễ, không đánh đổi chất lượng.

Giai đoạn 7 · Bài 16 cho bạn phần toán. Bài này cho bạn tầng huấn luyện. Một mô hình nháp tốt đáng giá gấp đôi hệ số tăng tốc so với một mô hình nháp rẻ tiền. EAGLE, EAGLE-2 và EAGLE-3 (Li et al., 2024–2025) đã biến "nháp = phiên bản nhỏ hơn của cùng mô hình" thành một kỷ luật kỹ thuật chính xác. Các server suy luận production năm 2026 mặc định dùng EAGLE-3.

## Khái niệm

### Bất biến: lấy mẫu loại bỏ Leviathan

Gọi `p(t)` là phân phối của mô hình nháp cho token kế tiếp với một tiền tố nào đó, và `q(t)` là phân phối của bộ xác minh. Lấy mẫu một token nháp `d ~ p`. Chấp nhận với xác suất `min(1, q(d) / p(d))`. Nếu loại, lấy mẫu từ phân phối phần dư `(q - p)_+ / ||(q - p)_+||_1`. Các mẫu thu được có phân phối theo `q`. Điều này đúng bất kể `p` tệ đến đâu — càng tệ thì bạn càng hay loại, nhưng đầu ra vẫn chính xác.

Xếp `N` lần gọi như vậy liên tiếp, dùng một lượt forward của bộ xác minh trên `prefix + d_1 + ... + d_N`. Bộ xác minh trả về `q_1, q_2, ..., q_{N+1}` cùng lúc. Đi từ trái sang phải. Ở lần loại đầu tiên tại vị trí `j`, lấy mẫu từ `residual(q_j, p_j)` rồi dừng. Nếu được chấp nhận toàn bộ, lấy mẫu thêm một token thưởng từ `q_{N+1}`.

### Điều gì quyết định hệ số tăng tốc

Gọi `α` là tỉ lệ chấp nhận kỳ vọng cho mỗi token được nháp. Gọi `c = cost(draft) / cost(verifier)` là tỉ lệ chi phí. Số token được chấp nhận kỳ vọng cho mỗi lượt forward của bộ xác minh là:

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

Tổng thời gian thực kỳ vọng cho mỗi token được chấp nhận là `(N * c + 1) / E[accepted]`. Cực tiểu hoá nó theo `N` và bạn có điểm ngọt. Với `α = 0.8, c = 0.05`: `N` tối ưu khoảng 5–7, hệ số tăng tốc là 3.2×. Với `α = 0.95, c = 0.02`: `N` tối ưu khoảng 8–10, hệ số tăng tốc chạm 5×.

Đòn bẩy lớn nhất là `α`. Đi từ `α = 0.6` (nháp nguyên bản) lên `α = 0.9` (EAGLE-3) với `N = 5` cố định đưa bạn từ 2.2 token được chấp nhận kỳ vọng mỗi lượt forward của bộ xác minh lên 4.1. Gần gấp đôi thông lượng với cùng một bộ xác minh.

### Chặng tiến hoá hai năm

**Speculative nguyên bản (Leviathan, 2023).** Mô hình nháp là một LLM nhỏ hơn, huấn luyện độc lập, cùng họ. Dễ nối vào, `α ≈ 0.6`, hệ số tăng tốc tốt nhất khoảng 2×.

**EAGLE-1 (Li et al., 2024).** Mô hình nháp là một transformer tí hon — thường một hoặc hai lớp — nhận hidden state lớp cuối của bộ xác minh làm đầu vào và dự đoán thẳng token kế tiếp. Vì mô hình nháp thấy được biểu diễn đặc trưng của bộ xác minh, phân phối của nó gần với phân phối của bộ xác minh hơn nhiều. `α` leo lên 0.7–0.8.

**EAGLE-2 (Li et al., 2024).** Thêm draft tree (cây nháp) động: thay vì đề xuất một dãy `N` token duy nhất, đề xuất một cây ứng viên nhỏ, chấm điểm từng ứng viên bằng bộ xác minh trong một lượt forward (tree attention), rồi đi theo đường có xác suất cao nhất. Độ dài nháp trở nên thích ứng theo từng bước. `α` trên mỗi token của đường được chấp nhận vượt 0.85.

**EAGLE-3 (Li et al., 2025, NeurIPS).** Thêm hai thay đổi. Thứ nhất, bỏ hẳn loss dự đoán đặc trưng — EAGLE-1/2 huấn luyện mô hình nháp khớp hidden state của bộ xác minh, và điều đó giới hạn mức dữ liệu còn giúp được. EAGLE-3 huấn luyện trực tiếp trên dự đoán token. Thứ hai, training-time test (TTT): trong lúc huấn luyện mô hình nháp, đưa chính các dự đoán trước đó của nó trở lại làm đầu vào qua nhiều bước, đúng như cách nó chạy khi suy luận. Điều này căn khớp phân phối huấn luyện và phân phối kiểm tra, đồng thời chặn tích luỹ sai số. Hệ số tăng tốc đo được: tới 6.5× trên chat, thông lượng cải thiện 38% ở batch 64 trong SGLang trên H100.

### Cuộn KV cache

Verification (xác minh) mở rộng KV cache của bộ xác minh thêm `N` mục trong một lượt. Nếu bị loại tại vị trí `j`, nội dung cache sau vị trí `j-1` là sai. Hai cách cài đặt phổ biến: ghi vào buffer tạm rồi commit khi được chấp nhận (vLLM, TensorRT-LLM), hoặc giữ một KV cache vật lý cộng một độ dài logic và cắt bớt khi bị loại. Cách nào thì chi phí cuộn lại cũng chỉ là vài byte mỗi lớp mỗi head, không đáng kể so với chi phí lượt forward.

Với tìm kiếm cây của EAGLE-2, bộ xác minh chạy attention với mask phi nhân quả tôn trọng cấu trúc cây. Phần kỹ thuật khá rườm rà nhưng phép tính chỉ là một lệnh flash-attention tiêu chuẩn với mask tuỳ biến.

### Kiến trúc nháp trong năm 2026

| Chiến lược | Loại nháp | `α` | Hệ số tăng tốc | Chi phí huấn luyện |
|----------|-----------|-----|---------|---------------|
| Nguyên bản | LLM nhỏ riêng biệt | 0.55-0.70 | 1.8-2.3× | Không (dùng lại mô hình nhỏ sẵn có) |
| Medusa | Thêm các đầu LM trên bộ xác minh | 0.65-0.75 | 2-3× | ~1B token SFT |
| EAGLE-1 | Transformer 1 lớp trên hidden state | 0.70-0.80 | 2.5-3× | ~60B token |
| EAGLE-2 | EAGLE-1 + cây nháp động | 0.80-0.88 | 3-4× | ~60B token |
| EAGLE-3 | feature fusion (hợp nhất đặc trưng) đa lớp + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B token |
| Lookahead | Không nháp (lặp Jacobi) | N/A | 1.3-1.6× | Không |

Trong production năm 2026: vLLM và SGLang mặc định dùng EAGLE-3 khi có, nếu không thì EAGLE-2. TensorRT-LLM có đường Medusa nhanh nhất cho các mô hình công khai của Meta và NVIDIA. llama.cpp dùng nháp nguyên bản cho triển khai CPU.

```figure
l5-spec-decode-eagle
```

## Tự tay làm

Xem `code/main.py`. Đây là vòng lặp suy đoán Leviathan đầy đủ với mọi mảnh: nháp N token, lượt chạy song song của bộ xác minh, loại theo từng vị trí, lấy mẫu phần dư, token thưởng, cuộn KV, và kiểm chứng thực nghiệm rằng phân phối đầu ra khớp với lấy mẫu trực tiếp từ `q`.

### Bước 1: quy tắc loại bỏ

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### Bước 2: phân phối phần dư

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### Bước 3: một bước suy đoán đầy đủ

Hàm `spec_step` nháp `N` token từ `p`, rồi xác minh tất cả trong một phép tính `q` song song. Với mỗi token được nháp, nó áp quy tắc loại bỏ, và ở lần loại đầu tiên nó lấy mẫu bản sửa từ phần dư. Nếu tất cả đều được chấp nhận, nó phát một token thưởng từ `q_{N+1}`.

### Bước 4: ghi sổ cuộn KV

Bộ mô phỏng theo dõi một `kv_length` logic cho mỗi worker. Khi chấp nhận `k` token nháp, `kv_length += k`. Khi bị loại tại vị trí `j`, cache đã ghi vượt qua `j`, nhưng độ dài logic được đặt thành `prefix_length + j + 1` — một vị trí sau token sửa. Các lần đọc sau đó cắt về độ dài logic.

### Bước 5: phép kiểm tra Leviathan

Chạy 50,000 bước suy đoán. Đếm phân phối thực nghiệm của các token được chấp nhận. So với 50,000 mẫu lấy trực tiếp từ `q`. Thống kê chi-square phải thấp hơn hẳn giá trị tới hạn. Định lý đúng trong thực tế.

### Bước 6: hệ số tăng tốc so với α

Quét chất lượng nháp bằng cách đẩy `p` ra xa `q` ở các biên độ khác nhau. Đo `α`, rồi vẽ số token kỳ vọng mỗi lần gọi bộ xác minh như một hàm của `α` và `N`. Code in ra một bảng cho thấy chất lượng nháp cỡ EAGLE-3 (`α ≈ 0.9`) mở ra 4–5 token mỗi lần gọi bộ xác minh.

## Áp dụng

`vllm serve` ở mức production với EAGLE-3:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang với EAGLE-3 ở batch 64 trên H100: thông lượng cao hơn khoảng 1.38× so với giải mã nguyên bản ở batch 64, theo bài báo EAGLE-3.

Khi nào nên dùng giải mã suy đoán:

- Mọi workload chat tương tác, nơi độ trễ p50 quan trọng hơn thông lượng đỉnh.
- Sinh code và đầu ra có cấu trúc (JSON, SQL). `α` trên 0.9 vì phân phối đích rất dễ dự đoán.
- Sinh văn bản dài (hàng nghìn token). Hệ số tăng tốc được phân bổ dần vẫn tiếp tục sinh lời.

Khi nào không nên:

- Mô hình rất nhỏ (< 3B). Mô hình nháp không rẻ hơn bộ xác minh là mấy.
- Triển khai CPU batch-1 tí hon. Chi phí bộ nhớ của mô hình nháp có thể không đáng.
- Lấy mẫu sáng tạo ở nhiệt độ rất cao, nơi `α` sụp.

## Đóng gói

Bài này sinh ra `outputs/skill-eagle3-tuner.md`. Cho một workload suy luận (mô hình, batch size, độ trễ mục tiêu, profile tác vụ), nó gợi ý một chiến lược giải mã suy đoán và các tham số tinh chỉnh (họ nháp, `N`, độ sâu cây, chuyển đổi theo nhiệt độ).

## Bài tập

1. Chạy `code/main.py`. Xác nhận thống kê chi-square trong phép kiểm tra phân phối Leviathan vẫn dưới giá trị tới hạn 95% trên 50,000 mẫu.

2. Quét `N` từ 1 đến 10 với `α` giữ ở 0.9 và `c` giữ ở 0.04. Vẽ số token kỳ vọng mỗi lần gọi bộ xác minh và thời gian thực tế mỗi token. Tìm `N` cực tiểu hoá thời gian thực. Giải thích hình dạng của đường cong.

3. Sửa code để mô phỏng tìm kiếm cây của EAGLE-2: ở mỗi bước, mô hình nháp đề xuất một cây có dạng `[2, 2, 2]` (tám đường ứng viên). Bộ xác minh chạy một lần, và đường được chấp nhận có xác suất cao nhất thắng. Tính `α` trên mỗi lá và tổng token mỗi lần gọi bộ xác minh. So với spec-decoding dạng chuỗi tuyến tính ở cùng mức tính toán.

4. Cài đặt bộ mô phỏng cuộn KV theo batch cho hai chuỗi đồng thời. Chuỗi A được chấp nhận toàn bộ token nháp; chuỗi B bị loại tại vị trí 2. Chứng minh rằng `kv_length` đúng được cập nhật cho từng chuỗi và không có công việc nào bị bỏ phí.

5. Đọc Mục 4 của bài báo EAGLE-3 (Training-Time Test). Giải thích trong hai câu vì sao huấn luyện nháp ngây thơ không có TTT lại bị exposure bias, và vì sao đưa chính dự đoán của mô hình nháp vào lúc huấn luyện lại sửa được điều đó. Nối điều này với tài liệu scheduled sampling trong seq2seq.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Leviathan rule | "min(1, q chia p)" | Chấp nhận/từ chối kiểu Bernoulli với xác suất `min(1, q(d)/p(d))`, bảo toàn chính xác phân phối của bộ xác minh khi bạn lấy mẫu từ phần dư lúc bị loại |
| Residual distribution | "(q trừ p) cộng, chuẩn hoá" | `(q - p)_+` kẹp ở 0 và chuẩn hoá lại — phân phối đúng để lấy mẫu khi bị loại |
| Acceptance rate α | "mô hình nháp đúng bao nhiêu lần" | Xác suất thành công Bernoulli kỳ vọng trên mỗi token theo quy tắc loại bỏ; chi phối toàn bộ phép toán hệ số tăng tốc |
| EAGLE-1 | "nháp từ hidden state" | Mô hình nháp transformer tí hon, điều kiện hoá trên hidden state lớp cuối của bộ xác minh (Li et al., 2024) |
| EAGLE-2 | "cây nháp động" | EAGLE-1 cộng một cây các phần nối tiếp ứng viên, chấm điểm bằng tree attention trong một lượt của bộ xác minh |
| EAGLE-3 | "kiểm tra lúc huấn luyện" | Bỏ loss dự đoán đặc trưng, huấn luyện trên dự đoán token trực tiếp với mô hình nháp được đưa chính đầu ra của nó vào trong lúc huấn luyện |
| Training-time test (TTT) | "sửa exposure bias" | Chạy mô hình nháp tự hồi quy trong lúc huấn luyện để phân phối đầu vào lúc huấn luyện và lúc kiểm tra khớp nhau — tương ứng trực tiếp với scheduled sampling |
| KV rollback | "hoàn tác token nháp bị loại" | Ghi sổ đưa KV cache của bộ xác minh về độ dài tiền tố được chấp nhận sau khi bị loại |
| Bonus token | "cái miễn phí" | Khi cả `N` token nháp được chấp nhận, lấy mẫu thêm một token từ `q_{N+1}` mà không tốn thêm chi phí bộ xác minh |
| Tree attention | "xác minh nhiều ứng viên cùng lúc" | Attention với mask phi nhân quả tôn trọng cấu trúc của cây nháp; tính `q_i` cho mọi nút trong cây trong một lượt forward |

## Đọc thêm

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — bài báo nền tảng và định lý tương đương
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — công trình độc lập ra đời cùng thời với một chứng minh gọn gàng
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1, mô hình nháp điều kiện hoá trên hidden state
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — tìm kiếm cây động
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — mặc định production của năm 2026
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — cách tiếp cận thay thế không cần mô hình nháp
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) — tài liệu tham chiếu production chuẩn với mọi chiến lược được nối sẵn

---

[← Bài trước: Mô hình mở: Điểm qua kiến trúc](../../14-open-models-architecture-walkthroughs/docs/vi.md) · [Bài sau: Attention sai phân (V2) →](../../16-differential-attention-v2/docs/vi.md)
