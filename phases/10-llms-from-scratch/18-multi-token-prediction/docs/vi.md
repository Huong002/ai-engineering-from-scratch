# Dự đoán nhiều token (MTP)

> Mọi LLM tự hồi quy từ GPT-2 đến Llama 3 đều huấn luyện với một mất mát cho mỗi vị trí: dự đoán token kế tiếp. DeepSeek-V3 thêm mất mát thứ hai cho mỗi vị trí: dự đoán token sau đó. 14B tham số dư ra (trên mô hình 671B) được chưng cất ngược vào mô hình chính qua dòng gradient, và các head MTP đã huấn luyện được tái dụng lúc suy luận thành draft (nháp) cho speculative decoding (giải mã suy đoán), với acceptance (chấp nhận) trên 80%. Throughput (thông lượng) sinh token đạt 1.8× mà không tốn thêm gì. Bài này tự tay dựng module MTP tuần tự từ báo cáo kỹ thuật DeepSeek, tính mất mát và cách bố trí tham số của head dùng chung, rồi giải thích vì sao MTP giữ được chuỗi nhân quả còn MTP song song nguyên bản của Gloeckle et al. thì phá vỡ nó.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 10, Bài 04 (huấn luyện trước một mini GPT), Giai đoạn 10, Bài 15 (giải mã suy đoán)
**Time:** ~60 phút

## Mục tiêu học tập

- Phát biểu mục tiêu huấn luyện MTP và suy ra mất mát kết hợp qua các độ sâu dự đoán.
- Giải thích khác biệt giữa các head MTP song song của Gloeckle et al. (2024) và các module MTP tuần tự của DeepSeek-V3, và vì sao thiết kế tuần tự giữ được chuỗi nhân quả.
- Tính chi phí tham số và bộ nhớ khi thêm module MTP vào một lần huấn luyện trước.
- Tự tay cài đặt một module MTP: embedding dùng chung, transformer block riêng cho từng độ sâu, phép chiếu, và đầu dự đoán (prediction head) dùng chung.

## Vấn đề

Dự đoán token kế tiếp là mục tiêu huấn luyện chuẩn của LLM. Mỗi hidden state được giám sát để dự đoán đúng một thứ: token ngay sau đó. Tín hiệu đó yếu đến bất ngờ. Phần lớn thông tin trong một chuỗi vượt ra ngoài một token — cấu trúc, mạch lạc, tính đúng sự thật, mạch số học. Mô hình phải học những thứ đó bằng cách tích lũy rất nhiều tín hiệu một token qua hàng nghìn tỉ token.

MTP đặt câu hỏi: nếu mỗi hidden state được giám sát để dự đoán nhiều token tương lai cùng lúc thì sao? Gloeckle et al. (Meta, 2024) cho thấy điều đó có ích. Cách cài đặt của họ gắn nhiều head đầu ra độc lập lên trên backbone, mỗi head dự đoán một độ lệch khác nhau. Song song, đơn giản, nhưng các head nhìn cùng một hidden state mà không có bước tinh chỉnh phân cấp nào — và các dự đoán không nối tiếp nhau theo nhân quả, nên không dùng được cho giải mã suy đoán.

DeepSeek-V3 (tháng 12 năm 2024) thiết kế lại MTP thành các module tuần tự giữ chuỗi nhân quả ở từng độ sâu dự đoán. Mô hình dự đoán `t+1` từ `h_i^(0)`, rồi dự đoán `t+2` từ hidden state mới `h_i^(1)` — kết hợp `h_i^(0)` với embedding `E(t+1)` — và cứ thế tiếp tục. Mỗi độ sâu là một transformer block nhỏ riêng. Embedding dùng chung và đầu dự đoán dùng chung giữ chi phí tham số ở mức khiêm tốn. Ở quy mô DeepSeek-V3, 14B tham số thêm nằm rải trong các module MTP, cộng lên 671B trọng số của mô hình chính. Khoản 2% đó mua được tín hiệu huấn luyện dày hơn VÀ một bộ nháp giải mã suy đoán dựng sẵn lúc suy luận.

Bài này tự tay dựng một module MTP và mất mát D độ sâu. Phần toán gọn gàng. Phần cài đặt khoảng 150 dòng.

## Khái niệm

### Công thức MTP tuần tự

DeepSeek-V3 thêm `D` module MTP lên trên mô hình chính. Mỗi module `k` (với `k = 1..D`) dự đoán token ở độ sâu `k` — tức `t_{i+k}` khi biết tiền tố tính đến vị trí `i`.

Module `k` gồm:

- Một transformer block `T_k` với attention và MLP riêng.
- Một ma trận chiếu `M_k` kết hợp hidden state của độ sâu trước với embedding của token đúng ở độ sâu kế tiếp.
- Embedding dùng chung `E` (giống mô hình chính).
- Đầu ra dùng chung `Out` (giống mô hình chính).

Khi huấn luyện, với tiền tố tính đến vị trí `i`, hidden state ở từng độ sâu là:

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

Dự đoán ở từng độ sâu là:

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

Mất mát ở từng độ sâu là cross-entropy so với `t_{i+k}` đúng:

```
L_k = CE(logits_{i+k}, t_{i+k})
```

Mất mát kết hợp qua các độ sâu:

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` là hệ số trọng số nhỏ — DeepSeek-V3 dùng 0.3 cho 10% đầu của quá trình huấn luyện và 0.1 sau đó. Tổng mất mát huấn luyện là `L_main + L_MTP`.

### Vì sao tuần tự chứ không song song

MTP song song nguyên bản của Gloeckle có D head đầu ra, mỗi head áp trực tiếp lên `h_i^(0)`. Mỗi head dự đoán `t_{i+k}` từ cùng một hidden state của backbone. Huấn luyện vẫn ổn, nhưng các dự đoán không lấy nhau làm điều kiện. Bạn không thể dùng đầu ra của `head_1` để giúp `head_2` — các head kích hoạt song song.

Thiết kế tuần tự của DeepSeek-V3 dựng `h_i^(k)` từ `h_i^(k-1)` cộng embedding thật của token kế tiếp `E(t_{i+k})`. Điều đó giữ được chuỗi nhân quả: để dự đoán `t_{i+k+1}`, module ở độ sâu `k+1` thấy được thứ đã có ở `t_{i+k}`. Về cấu trúc, việc này giống hệt cách một decoder tự hồi quy tiêu thụ đầu ra của chính nó — khiến các module MTP dùng thẳng được làm bộ nháp cho giải mã suy đoán.

Lúc suy luận: đưa `h_i^(k-1)` và `t_{i+k}` vừa nháp vào module `k+1`, nhận dự đoán cho `t_{i+k+1}`. Lặp lại. Đó chính xác là kiểu nháp của EAGLE, dùng module MTP đã huấn luyện làm mạng nháp. DeepSeek-V3 báo cáo tỉ lệ chấp nhận trên 80% ở module MTP đầu tiên và tốc độ nhanh hơn ~1.8×.

### Tính toán tham số

Với mô hình có hidden `h` và từ vựng `V`:

- Mô hình chính: hàng tỉ tham số, cộng một đầu ra kích thước `V * h`.
- Đầu ra dùng chung: tái dùng head của mô hình chính. Không thêm tham số.
- Embedding dùng chung: tái dùng embedding của mô hình chính. Không thêm tham số.
- Mỗi module MTP:
  - Phép chiếu `M_k`: `(2h) * h = 2h^2`.
  - Transformer block `T_k`: attention (`4h^2` cho MHA) cộng MLP (thường `8h^2` cho SwiGLU với tỉ lệ 8/3). Khoảng `12h^2` mỗi block.

Tổng thêm cho mỗi module: `~14h^2`. Với `h = 7168` của DeepSeek-V3, D = 1 module: `~14 * 7168^2 = ~720M` tham số trên giấy. DeepSeek-V3 báo cáo 14B — chênh lệch chủ yếu vì các lớp expert trong module MTP cũng là MoE.

### Phần thưởng của giải mã suy đoán

Trong lúc huấn luyện trước, các module MTP làm huấn luyện chậm đi khoảng 10% (tính forward nhiều hơn, thêm mất mát). Lợi ích gấp đôi:

1. Tín hiệu huấn luyện dày hơn. Mỗi hidden state thấy D+1 mục tiêu giám sát. Tác động đo được trên MMLU, GSM8K, MATH, HumanEval: cải thiện đều vài điểm phần trăm trong các thử nghiệm ablation của DeepSeek-V3.

2. Bộ nháp giải mã suy đoán miễn phí lúc suy luận. Module MTP đã được huấn luyện để dự đoán vài token kế tiếp. Tái dụng làm mạng nháp, nó cho tỉ lệ chấp nhận trên 80%. Ở mức đó, giải mã suy đoán với N=3 hoặc N=5 cho thông lượng 1.8×. Chi phí 10% lúc huấn luyện hoàn vốn ngay lần đầu bạn chạy suy luận.

### Liên hệ với EAGLE

EAGLE huấn luyện một mô hình nháp nhỏ RIÊNG sau khi huấn luyện trước. MTP nướng luôn phần nháp vào quá trình huấn luyện trước. Hai cách hội tụ về tỉ lệ chấp nhận tương tự nhau nhưng qua hai pipeline khác nhau:

| Khía cạnh | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| Thời điểm huấn luyện | Sau huấn luyện trước | Trong huấn luyện trước |
| Tương thích ngược với trọng số có sẵn | Có | Không (phải huấn luyện lại) |
| Tham số nháp | 1-2 lớp transformer | 1 transformer block + phép chiếu |
| Tỉ lệ chấp nhận | 0.88-0.92 | 0.80+ ở độ sâu 1 |
| Lợi ích ngoài tốc độ | Chỉ giải mã suy đoán | Tín hiệu huấn luyện dày hơn + nhanh hơn |

```figure
multi-token-predict
```

## Tự tay làm

`code/main.py` dựng trọn một module MTP từ đầu đến cuối: embedding dùng chung, phép chiếu, transformer block, đầu ra dùng chung. Sau đó nó tính mất mát cross-entropy theo từng độ sâu trên một chuỗi tổng hợp ngắn và in số tham số theo từng thành phần. Từ vựng đồ chơi 32 token giữ cho các con số dễ đọc.

### Bước 1: bảng embedding dùng chung

Một bảng `vocab_size x hidden` duy nhất được mô hình chính VÀ mọi module MTP ở mọi độ sâu dùng chung. Không phải bản sao thứ hai — đúng nghĩa là cùng một tensor.

### Bước 2: phép kết hợp theo độ sâu

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

DeepSeek-V3 thật nối hai vector đã RMSNorm thành `[2h]` rồi chiếu bằng ma trận `h x 2h`. Bản đồ chơi dùng phép cộng vector cho gọn theo stdlib.

### Bước 3: transformer block ở độ sâu k

Self-attention cộng MLP. Trong bản đồ chơi, một block linear attention một lớp và một MLP SwiGLU giữ cấu trúc hiện rõ mà không cần numpy.

### Bước 4: đầu ra dùng chung

Tái dùng phép chiếu đầu ra của mô hình chính. Logits trên toàn từ vựng.

### Bước 5: mất mát theo độ sâu

Cross-entropy của softmax(logits) so với token đúng ở độ lệch `k`. Gộp qua các độ sâu bằng hệ số tỉ lệ `lambda / D`.

### Bước 6: tính toán tham số

In tổng số tham số, số tham số dùng chung (embedding, head), và số tham số thêm cho mỗi module. Hiển thị tỉ lệ phần MTP thêm so với kích thước mô hình chính.

## Áp dụng

MTP đã được tích hợp vào DeepSeek-V3 (tháng 12 năm 2024) và dòng DeepSeek-R1. Lúc suy luận:

- Stack phục vụ của chính DeepSeek dùng module MTP làm bộ giải mã suy đoán ngay khi cài.
- vLLM và SGLang có đường tích hợp cho MTP của DeepSeek-V3 tính đến tháng 4 năm 2026.
- Hướng dẫn ROCm SGLang của AMD trình bày một cấu hình giải mã suy đoán MTP cụ thể, đo được tốc độ nhanh hơn 1.8× trên checkpoint V3.

Khi nào nên dùng MTP trong một lần huấn luyện trước mới:

- Bạn kiểm soát toàn bộ pipeline huấn luyện trước và muốn tích sẵn tín hiệu huấn luyện dày hơn.
- Bạn biết sẽ phục vụ mô hình ở quy mô lớn và muốn có giải mã suy đoán miễn phí.
- Kích thước hidden của bạn ít nhất 4096. Ở quy mô 1B, chi phí thêm gây hại nhiều hơn lợi ích mang lại.

Khi nào không nên:

- Khi fine-tune một mô hình dense đã huấn luyện trước. Module MTP không được huấn luyện.
- Mô hình nghiên cứu mà bạn muốn một baseline sạch để so sánh. MTP thay đổi kiến trúc.

## Đóng gói

Bài này tạo ra `outputs/skill-mtp-planner.md`. Cho một đặc tả lần huấn luyện trước (kích thước mô hình, dữ liệu, compute), nó trả về kế hoạch tích hợp MTP: số độ sâu D, lịch `lambda`, chi phí bộ nhớ, và cách đấu dây giải mã suy đoán lúc suy luận.

## Bài tập

1. Chạy `code/main.py`. Chứng minh mất mát theo từng độ sâu giảm đơn điệu khi tín hiệu tổng hợp mạnh lên. Sửa phần tổng hợp để dùng một mẫu cố định và kiểm tra cả mất mát độ sâu 1 lẫn độ sâu 2 đều hội tụ.

2. Tính chi phí tham số cho một mô hình dense 70B (hidden 8192, 80 lớp) với D=1 module MTP. So sánh với con số 14B mà DeepSeek-V3 báo cáo. Giải thích vì sao con số của DeepSeek cao hơn: transformer block trong MTP thừa hưởng đúng cấu trúc MoE đó, làm phồng số tham số mỗi module.

3. Cài D=2 trong bản đồ chơi: thêm module MTP thứ hai nhận h^(1) và dự đoán `t_{i+2}`. Kiểm tra mất mát kết hợp và cách tính tham số khớp với phương trình 19-21 trong bài báo DeepSeek.

4. Chuyển bản đồ chơi sang MTP song song (kiểu Gloeckle): thêm D head đầu ra lên trên hidden state chính, mỗi head dự đoán một độ lệch khác nhau. Đo xem mất mát theo từng độ sâu so với bản tuần tự thế nào trên cùng tín hiệu tổng hợp. Bản tuần tự phải cho mất mát độ sâu k thấp hơn với k > 1 vì nó lấy các dự đoán trung gian làm điều kiện.

5. Dùng module MTP đã huấn luyện làm bộ nháp kiểu EAGLE: gọi module k để đề xuất `t_{i+k}` lúc suy luận. Đo tỉ lệ chấp nhận của các token nháp đó so với dự đoán của mô hình chính trên một chuỗi giữ lại. Nếu đạt trên 50% ở bản đồ chơi, bạn đã tái hiện được tính chất MTP-làm-nháp đã quan sát trong thực nghiệm.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| MTP module | "Khối mất mát phụ (auxiliary loss)" | Một transformer block nhỏ cộng phép chiếu, dự đoán token cách mô hình chính `k` vị trí về phía trước |
| Prediction depth | "Độ lệch nào" | Số nguyên `k` sao cho module `k` dự đoán `t_{i+k}` từ tiền tố tính đến vị trí `i` |
| Parallel MTP | "Kiểu Gloeckle" | D head độc lập trên cùng một hidden state của backbone, không có chuỗi điều kiện |
| Sequential MTP | "Kiểu DeepSeek-V3" | Mỗi module lấy hidden state của độ sâu trước cộng embedding của token kế tiếp làm điều kiện; giữ được chuỗi nhân quả |
| Shared output head | "Dùng lại head chính" | Các module MTP gọi LM head của mô hình chính, không dùng phép chiếu đầu ra riêng |
| Shared embedding | "Dùng lại bảng chính" | Cùng một bảng embedding từ vựng được dùng ở mọi nơi; không nhân bản tham số |
| Projection matrix M_k | "Gộp hidden + token kế" | Một lớp tuyến tính `h x 2h` gộp hidden state trước đó và embedding của token mục tiêu thành đầu vào cho độ sâu kế tiếp |
| Joint loss L_MTP | "Trung bình các mất mát phụ" | Trung bình cộng của các mất mát cross-entropy theo độ sâu, nhân với `lambda` |
| Acceptance rate at depth 1 | "Bao lâu thì nháp MTP đúng" | Tỉ lệ mà dự đoán top-1 của module MTP với D=1 trùng với dự đoán top-1 của mô hình chính; trên 80% ở DeepSeek-V3 |
| Lambda weighting | "Tầm quan trọng của mất mát phụ" | Hệ số tỉ lệ theo từng độ sâu; 0.3 lúc bắt đầu huấn luyện, 0.1 về sau trên DeepSeek-V3 |

## Đọc thêm

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — mô tả đầy đủ về MTP tuần tự (Mục 2.2), gồm các phương trình mất mát kết hợp và tốc độ nhanh hơn 1.8× lúc suy luận
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — baseline MTP song song mà thiết kế của DeepSeek cải tiến
- [Thẻ mô hình DeepSeek-V3 trên Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — tổng 685B (671B chính + 14B MTP), ghi chú triển khai
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — khung giải mã suy đoán mà MTP khớp vào
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — kiến trúc nháp 2025 của EAGLE, đối thủ mà MTP cạnh tranh

---

[← Bài trước: Native Sparse Attention (DeepSeek NSA)](../../17-native-sparse-attention/docs/vi.md) · [Bài sau: Song song DualPipe →](../../19-dualpipe-parallelism/docs/vi.md)
