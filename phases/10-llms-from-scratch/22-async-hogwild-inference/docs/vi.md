# Suy luận bất đồng bộ và Hogwild!

> Giải mã suy đoán (Giai đoạn 10 · 15) song song hoá token bên trong một chuỗi. Các framework đa tác tử song song hoá trên toàn bộ chuỗi nhưng buộc phải phối hợp tường minh (bỏ phiếu, chia bài toán con). Hogwild! Inference (Rodionov et al., arXiv:2504.06261) làm điều khác: chạy N bản sao của cùng một LLM song song trên một key-value cache CHUNG. Mỗi worker thấy token mà mọi worker khác sinh ra ngay lập tức. Các mô hình suy luận hiện đại — QwQ, DeepSeek-R1 — có thể tự phối hợp qua cache chung đó mà không cần fine-tuning. Cách tiếp cận này còn mang tính thử nghiệm nhưng nó mở ra một trục hoàn toàn mới của song song hoá suy luận, nằm trực giao với spec decode. Bài này cài đặt một trình mô phỏng Hogwild! hai worker bằng Python thuần và giải thích vì sao sự cộng tác qua cache chung lại nảy sinh từ năng lực suy luận sẵn có của mô hình.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 10, Bài 12 (tối ưu suy luận), Giai đoạn 10, Bài 15 (giải mã suy đoán)
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả ba kiến trúc song song LLM phổ biến (bỏ phiếu, bài toán con, Hogwild!) và nêu mỗi kiến trúc nhắm tới bài toán nào.
- Phát biểu thiết lập cốt lõi của Hogwild!: nhiều worker, một KV cache chung, phối hợp nảy sinh qua tự nhắc (self-prompting).
- Tính mức tăng tốc thời gian thực của Hogwild! như hàm của số worker `N`, độ song song cấp bài toán `p`, và chi phí phối hợp `c`.
- Cài đặt một trình mô phỏng Hogwild! hai worker trên bài toán đồ chơi và quan sát sự phân chia công việc nảy sinh.

## Vấn đề

Các LLM hiện đại giải bài toán khó bằng cách sinh ra những chuỗi suy luận dài — 5000 token lập luận từng bước là chuyện thường, hàng chục nghìn token cũng xảy ra với các bài toán sâu. Với tốc độ giải mã 35 token/giây trên một mô hình 70B, 50k token mất 24 phút. Mô hình như thế không thể tương tác được.

Giải mã suy đoán (Giai đoạn 10 · 15) giúp bạn tăng thông lượng (throughput) 3-5x nhờ song song hoá bên trong một chuỗi. Vượt qua mức đó, phụ thuộc tuần tự của giải mã tự hồi quy chính là trần cứng. Mỗi token mới phụ thuộc vào mọi token trước đó.

Câu hỏi hiển nhiên: liệu có thể song song hoá trên nhiều chuỗi? Chạy nhiều bản sao của cùng mô hình trên cùng bài toán, để chúng hợp tác, để chúng chia việc?

Các hướng nghiên cứu trước: ensemble bỏ phiếu (chạy N mô hình, chọn câu trả lời đa số), tree-of-thought (phân nhánh các đường suy luận rồi kết hợp lại), và các framework đa tác tử (giao mỗi tác tử một bài toán con, dùng một bộ điều phối). Tất cả đều có ích trong những miền bài toán cụ thể. Tất cả cũng đều mang theo bộ máy phối hợp tường minh — luật bỏ phiếu, logic phân nhánh rồi cắt tỉa, giao thức nhắn tin giữa các tác tử.

Hogwild! Inference tiếp cận theo hướng khác. N worker chia sẻ một KV cache duy nhất. Mỗi worker thấy token mà mọi worker khác sinh ra ngay lập tức, như thể chúng nằm trong ngữ cảnh của chính nó. Các worker — không cần huấn luyện hay fine-tuning — tự tìm ra cách chia việc. Các mô hình suy luận hiện đại (QwQ, DeepSeek-R1, chế độ suy luận của dòng Claude) đọc được cache chung và nói những câu như "Tôi thấy worker 2 đã xử lý trường hợp cơ sở rồi, nên tôi sẽ làm bước quy nạp."

Mức tăng tốc phụ thuộc khối lượng công việc và vẫn còn thử nghiệm tính đến tháng 4 năm 2026. Nhưng ý tưởng này đáng biết vì nó mở ra một trục mới của song song hoá suy luận.

## Khái niệm

### Thiết lập

Khởi tạo N tiến trình worker, tất cả chạy cùng một LLM. Thay vì KV cache riêng cho từng worker, duy trì MỘT cache chung. Khi worker `i` sinh token `t_j`, token được ghi vào cache chung ở vị trí kế tiếp. Khi worker `k` thực hiện bước tiếp theo, nó đọc trạng thái hiện tại của cache (bao gồm mọi thứ mà cả N worker đã sinh ra cho tới lúc đó).

Ở thời điểm một bước, các worker tranh nhau ghi token. Không có chỉ số vị trí riêng cho từng worker — cache là một chuỗi đang lớn dần duy nhất. Thứ tự được quyết định bởi thời điểm ghi tới.

### Vì sao sự phối hợp nảy sinh

Các worker dùng chung một prompt. Thường là đại loại "Bạn là một trong N bản sao cùng làm việc trên bài toán này. Mỗi bản sao đọc bộ nhớ chung và thấy được những gì bản sao khác đã viết. Tránh làm trùng việc." Prompt cộng với cache chung là đủ. Các mô hình suy luận đọc cache, nhận ra phần nào của bài toán đã được thử, và (thường, nhưng không phải luôn luôn) chuyển sang phần chưa được khám phá.

Bài báo Hogwild! (Rodionov et al., 2025) ghi lại những quan sát như:

- Các worker lập kế hoạch và truyền đạt kế hoạch cho worker khác qua cache.
- Các worker phát hiện lỗi trong lập luận của worker khác và chỉ ra.
- Các worker thích ứng khi một kế hoạch thất bại và đề xuất phương án khác.
- Khi được nhắc kiểm tra trùng lặp, các worker phát hiện và chuyển hướng.

Không điều nào trong số này cần fine-tuning. Hành vi nảy sinh đến từ năng lực suy luận mà mô hình đã có sẵn.

### Cái tên

Tên bài báo chơi chữ từ Hogwild! SGD (Recht et al., 2011), một bộ tối ưu cập nhật bất đồng bộ (asynchronous), không khoá (lock-free). Sự tương tự: các worker bất đồng bộ của SGD đều ghi vào một vector tham số chung, nên mỗi worker có thể đọc trọng số cũ (stale weights); worker của Hogwild! Inference cũng vậy, đều ghi vào một KV cache chung. Cả hai dựa vào hội tụ (convergence) thực nghiệm thay vì các bảo đảm đồng bộ hoá.

### RoPE khiến việc này khả thi

Rotary Position Embeddings (RoPE, Su et al. 2021) mã hoá thông tin vị trí bằng phép quay trên các vector Q và K. Vì vị trí là các phép quay chứ không phải độ lệch gắn cứng, vị trí của một token có thể dịch chuyển mà không cần tính lại mục KV cache. Khi worker `i` ghi vào cache chung ở vị trí `p`, các worker khác đọc vị trí đó có thể dùng trực tiếp mục đã nằm trong cache — không cần quay lại.

Với mô hình dùng vị trí học được hoặc vị trí tuyệt đối, Hogwild! sẽ cần vô hiệu hoá cache sau mỗi lần ghi đồng thời. RoPE cho phép cache giữ được ổn định.

### Bài toán thời gian thực

Gọi `T_serial` là thời gian để một worker giải bài toán một mình. Gọi `p` là phần song song hoá được ở cấp bài toán. Gọi `c` là chi phí phối hợp mỗi bước (đọc cache đã mở rộng, quyết định ghi gì).

Thời gian một worker: `T_serial`.
Thời gian Hogwild! với N worker, nếu phối hợp miễn phí: `T_serial * ((1 - p) + p / N)`. Định luật Amdahl kinh điển.
Kèm chi phí phối hợp: `T_serial * ((1 - p) + p / N) + c * steps_per_worker`.

Để một worker còn hữu ích, `c` phải nhỏ so với thời gian giải mã mỗi bước. Trên các mô hình suy luận sinh 5k+ token, các worker có thể chịu hàng trăm token chi phí phối hợp mà vẫn có lợi. Với các tác vụ chat ngắn, phối hợp chiếm ưu thế và Hogwild! còn tệ hơn chạy tuần tự.

### Ví dụ cụ thể

Bài toán suy luận: 10k token chain-of-thought. Giả sử bài toán có `p = 0.7` nội dung song song hoá được (các chiến lược chứng minh khác nhau, các cách phân tích trường hợp khác nhau) và `c = 200` token chi phí phối hợp cho mỗi worker. Với `N = 4` worker:

- Thời gian tuần tự: 10000 bước giải mã.
- Thời gian Hogwild!: 10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 bước giải mã.
- Mức tăng tốc: 10000 / 5550 = 1.8x.

Con số này khiêm tốn. Nhưng với các bài toán suy luận dài hơn (50k token), chi phí phối hợp được khấu hao và mức tăng tốc vươn tới 2.5-3x. Hogwild! là tương đương trong suy luận của song song hoá cấp luồng ở một ngôn ngữ cho phép bạn viết code đa luồng một cách tự nhiên.

### Khi nào nên dùng Hogwild!

- Bài toán suy luận dài (hàng nghìn token) mà công việc có thể song song hoá thành các mục tiêu con độc lập.
- Các mô hình suy luận đã được huấn luyện để suy nghĩ từng bước. Mô hình không suy luận tự phối hợp kém.
- Triển khai trên một node với đủ VRAM để chứa cache chung cộng N tiến trình worker. Cache được chia sẻ, nhưng mỗi worker có bộ nhớ activation riêng.

### Khi nào không nên

- Chat tương tác ngắn. Chi phí phối hợp chiếm ưu thế.
- Các tác vụ không song song hoá được (một chứng minh tuyến tính, một lần biên dịch). N=1 là tối đa.
- Mô hình không suy luận. Không có phối hợp nào nảy sinh.
- Triển khai nhiều node. Cache chung cần đồng bộ hoá giữa các worker rất nhanh. Trong cùng một node thì ổn; xuyên node là thảm hoạ về độ trễ (latency).

### Tình trạng thử nghiệm

Tính đến tháng 4 năm 2026, Hogwild! là một phương pháp nghiên cứu với một cài đặt PyTorch mã nguồn mở. Chưa có ứng dụng nào trong môi trường sản xuất. Ba rào cản:

1. Quản lý KV cache chung giữa các tiến trình đồng thời là bài toán kỹ thuật không đơn giản.
2. Phối hợp nảy sinh phụ thuộc tác vụ; các bộ đánh giá vẫn đang được xây dựng.
3. Mức tăng tốc còn khiêm tốn so với những gì giải mã suy đoán đã mang lại, và hai cách có thể kết hợp nhưng phần kỹ thuật của sự kết hợp lại là một tầng nữa.

Đáng biết. Đáng thử nghiệm. Chưa đáng để đặt cược cả một sản phẩm.

```figure
continuous-batching
```

## Tự tay làm

`code/main.py` cài đặt một trình mô phỏng Hogwild! đồ chơi:

- Hai tiến trình worker, mỗi cái là một "LLM" tất định sinh ra một trong vài loại token (work-token, observe-token, coordinate-token) với xác suất đã biết.
- Một cache chung (chỉ là danh sách token) mà cả hai worker đều đọc và ghi.
- Một logic phối hợp đơn giản: khi một worker thấy worker kia đã sinh đủ token công việc trong một loại, nó chọn loại khác.

Trình mô phỏng chạy trong một ngân sách bước cố định và báo cáo:

- Tổng số work-token đã sinh.
- Tổng thời gian thực (số bước của worker).
- Mức tăng tốc hiệu dụng so với một worker.
- Vết ghi lại worker nào đã viết token nào.

### Bước 1: cache chung

Một danh sách mà cả hai worker cùng append vào. Trong một cài đặt thật cần khoá đơn giản (Python `threading.Lock`) để tránh tranh chấp dữ liệu (data race); ở đây ta mô phỏng bằng một biến đếm.

### Bước 2: vòng lặp worker

Mỗi worker, ở mỗi bước:

- Đọc cache chung hiện tại.
- Quyết định loại token sẽ ghi dựa trên những gì đã có.
- Ghi một token.

### Bước 3: heuristic phối hợp

Nếu loại X đã có K token trong cache và loại mà worker định ghi là X, worker chuyển sang loại Y. Đây là bản thay thế đồ chơi cho hành vi của mô hình suy luận: "nhận ra phần này đã được làm rồi, làm việc khác thay thế."

### Bước 4: đo mức tăng tốc

Chạy trình mô phỏng với N=1 worker và với N=2 worker, cùng tổng ngân sách bước. Đếm số work-token sinh ra. N=2 sẽ sinh được khoảng 1.5-1.8x work-token nhờ việc phân chia công việc do phối hợp.

### Bước 5: thử thách sự phối hợp

Giảm độ nhạy của heuristic phối hợp. Chạy lại. Quan sát rằng khi không có phối hợp tốt, N=2 sinh trùng lặp cùng các token và mức tăng tốc tụt xuống dưới 1. Điều này khớp với quan sát trong bài báo: thủ thuật chỉ hiệu quả nếu các worker có đủ năng lực suy luận để tự phối hợp.

## Áp dụng

Tính đến tháng 4 năm 2026, việc tích hợp Hogwild! vào sản xuất vẫn ở mức nghiên cứu. Cài đặt tham chiếu từ Yandex/HSE/IST dựa trên PyTorch và nhắm tới các thiết lập đa tiến trình trên một node với mô hình DeepSeek-R1 và QwQ.

Lộ trình áp dụng thực dụng:

1. Đo hồ sơ khối lượng công việc suy luận của bạn. Đo phần token mang tính khám phá (nhiều chiến lược, phân tích trường hợp, tìm kiếm) so với phần tuyến tính.
2. Nếu khám phá chiếm ưu thế, chạy một thử nghiệm Hogwild! hai worker. Đo mức cải thiện thời gian thực.
3. Nếu mức cải thiện dưới 1.3x, bạn đang ở vùng phối hợp chiếm ưu thế. Quay lại một worker.
4. Nếu mức cải thiện trên 1.5x, đẩy lên N=4 và đo lại. Lợi suất giảm dần thường xuất hiện quanh N=4-8.

Kết hợp với giải mã suy đoán: mỗi worker Hogwild! có thể độc lập dùng spec decode. Hai mức tăng tốc nhân với nhau (xấp xỉ), đưa 3x từ spec decode và 1.8x từ Hogwild! thành 5.4x hiệu dụng so với giải mã một worker thông thường.

## Đóng gói

Bài này tạo ra `outputs/skill-parallel-inference-router.md`. Cho trước một hồ sơ khối lượng công việc suy luận (ngân sách token, hồ sơ song song cấp bài toán, họ mô hình, mục tiêu triển khai), nó định tuyến giữa các chiến lược bỏ phiếu, tree-of-thought, đa tác tử, Hogwild!, và giải mã suy đoán.

## Bài tập

1. Chạy `code/main.py` với các thiết lập mặc định. Xác nhận cấu hình Hogwild! N=2 sinh nhiều work-token hơn mốc N=1 trong cùng thời gian thực.

2. Giảm độ mạnh của heuristic phối hợp (đặt `coordination_weight=0.1`). Chạy lại. Chứng minh mức tăng tốc sụp đổ. Giải thích vì sao: các worker làm trùng việc khi chúng không phối hợp được.

3. Tính mức tăng tốc Hogwild! kỳ vọng cho một bài toán suy luận 50k token với `p=0.8, c=500` và N=4 worker. Làm tương tự cho một tác vụ chat 1k token với `p=0.3, c=200` và N=4. Vì sao cái này thắng còn cái kia thua?

4. Đọc Mục 4 của bài báo Hogwild! (đánh giá sơ bộ). Xác định hai kiểu thất bại mà các tác giả báo cáo. Mô tả cách một prompt phối hợp tốt hơn có thể giảm nhẹ từng kiểu.

5. Kết hợp Hogwild! với giải mã suy đoán trong bài đồ chơi: mỗi worker dùng nội bộ spec-decode 2 token. Báo cáo mức tăng tốc nhân tính. Vấn đề ghi sổ nào nảy sinh khi hai worker cùng muốn mở rộng một tiền tố cache chung?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Hogwild! | "Worker song song, cache chung" | N bản sao của cùng một LLM chạy đồng thời với một KV cache chung; phối hợp nảy sinh qua tự nhắc |
| Shared KV cache | "Môi trường phối hợp" | Một bộ đệm KV đang lớn dần duy nhất mà mọi worker đọc và ghi; cho phép token hiện ra tức thì giữa các worker |
| Emergent coordination | "Không cần huấn luyện" | LLM có năng lực suy luận đọc được cache chung và chia việc mà không cần fine-tuning hay giao thức tường minh nào |
| Coordination overhead (c) | "Token dùng để định hướng" | Chi phí mỗi worker để đọc cache đã mở rộng và quyết định làm gì; phải nhỏ so với tổng thời gian giải mã |
| Parallelizable fraction (p) | "Phần chạy song song được" | Song song cấp bài toán: phần công việc không mang tính tuần tự nội tại |
| RoPE enables Hogwild! | "Vị trí quay bất biến với dịch chuyển" | Vì vị trí là các phép quay, ghi vào cache chung không cần tính lại các token trước đó |
| Voting ensemble | "Chạy N, chọn đa số" | Kiến trúc suy luận song song đơn giản nhất; hữu ích cho phân loại, kém hơn cho suy luận dạng dài |
| Tree of thought | "Phân nhánh và cắt tỉa" | Chiến lược suy luận khám phá nhiều nhánh rồi cắt tỉa; logic phối hợp tường minh |
| Multi-agent framework | "Giao bài toán con" | Mỗi tác tử nhận một vai trò; một bộ điều phối điều hành; chi phí giao thức lớn |

## Đọc thêm

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — bài báo Hogwild!, đánh giá sơ bộ trên QwQ và DeepSeek-R1
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — Hogwild! gốc, nguồn gốc của cái tên
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE, tính chất khiến suy luận với cache chung khả thi
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — chiến lược suy luận tree-of-thought mà Hogwild! nằm trực giao
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — giải mã suy đoán, kiểu song song bên trong một chuỗi mà Hogwild! kết hợp cùng
- [Cài đặt PyTorch tham chiếu của Hogwild!](https://github.com/eqimp/hogwild_llm) — nguồn chân lý duy nhất cho các thực nghiệm của bài báo

---

[← Bài trước: Jamba](../../21-jamba-hybrid-ssm-transformer/docs/vi.md) · [Bài sau: Giải mã suy đoán và EAGLE →](../../25-speculative-decoding/docs/vi.md)
