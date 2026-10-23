# Từ chatbot đến agent xử lý tác vụ dài

> Năm 2023, chatbot trả lời một câu hỏi trong một lượt. Năm 2026, mô hình hàng đầu thường chạy từ vài phút đến vài giờ cho một tác vụ. Benchmark Time Horizon 1.1 của METR (tháng 1/2026) ghi nhận Claude Opus 4.6 xử lý hơn 14 giờ công việc của chuyên gia với độ tin cậy 50%. Kể từ GPT-2, thời lượng này tăng gấp đôi khoảng mỗi bảy tháng. Mọi giả định dành cho chat một lượt — ngữ cảnh, độ tin cậy, cách thất bại, chi phí, khả năng quan sát — đều không còn đúng khi tác vụ kéo dài quá giờ ăn trưa.

**Type:** Learn
**Languages:** Python (stdlib, horizon-curve simulator)
**Prerequisites:** Giai đoạn 14, Bài 01 (Vòng lặp Agent)
**Time:** ~45 phút

## Vấn đề

Chatbot là một hàm không lưu trạng thái. Nó nhận prompt, trả lời rồi quên. Ngay cả hệ thống có RAG được xây dựng đến năm 2024 cũng hoạt động như vậy: lập kế hoạch trong một cửa sổ ngữ cảnh, thực hiện một hành động rồi trả kết quả.

Agent tự chủ khác về bản chất. Nó chạy theo vòng lặp và tự quyết định khi nào dừng. Trong lúc chạy, nó tiêu tốn tài nguyên — token thật, giờ GPU thật — và gây tác động thực ở các bước sau. Agent xử lý tác vụ dài khuếch đại mọi mặt: chi phí tăng, xác suất lỗi tăng theo từng bước, và khoảng cách giữa thứ ta đánh giá với thứ được triển khai ngày càng rộng.

Số liệu METR cho thấy rõ điều đó. Từ GPT-2 đến Claude Opus 4.6, time horizon (thời lượng tác vụ mà chuyên gia cần, còn mô hình hoàn thành với độ tin cậy 50%) tăng từ vài giây lên nửa ngày làm việc. Chu kỳ tăng gấp đôi gần bảy tháng. Nếu xu hướng kéo dài thêm một năm, horizon 50% sẽ đạt tác vụ nhiều ngày. Điều này khác hẳn những gì thời chatbot được thiết kế để xử lý.

## Khái niệm

### METR Time Horizon, trong một đoạn

METR (trước đây là ARC Evals) khớp một đường cong logistic giữa xác suất hoàn thành tác vụ và logarit thời gian chuyên gia cần để hoàn thành. Horizon là giao điểm của đường cong với mức xác suất 50%. Bộ tác vụ (HCAST, RE-Bench, SWAA) trải từ 1 phút đến hơn 8 giờ làm việc chuyên gia trong phần mềm, an ninh mạng, nghiên cứu ML và suy luận tổng quát. Kết quả là một con số gói gọn năng lực trong đơn vị dễ hiểu: “mô hình này làm được loại tác vụ mà chuyên gia cần X giờ”.

### Điều gì thực sự hỏng khi horizon tăng

- **Ngữ cảnh.** Lượt chạy 14 giờ tạo ra hàng trăm nghìn token gồm quan sát, đầu ra công cụ và dấu vết suy luận. Bạn không thể giữ toàn bộ lịch sử thô; bạn cần nén, checkpoint (điểm lưu trạng thái) và các tầng bộ nhớ (Giai đoạn 14 · 04-06).
- **Độ tin cậy.** Với một lượt, bạn đọc được toàn bộ câu trả lời. Với 1,000 lượt thì không. Việc rà soát chuyển từ “đọc đầu ra” sang “kiểm tra trajectory (chuỗi diễn tiến)”.
- **Cách thất bại.** Lượt chạy ngắn thất bại vì giới hạn năng lực. Lượt chạy dài còn có thể lệch hướng, mắc vòng lặp, reward hacking (lách mục tiêu thưởng) và hành xử khác giữa đánh giá với triển khai (xem bên dưới). Những lỗi này khó thấy cho đến khi tích tụ.
- **Chi phí.** Một lượt Claude Opus 4.6 tự chạy 14 giờ và dùng công cụ tối đa có thể ngốn ngân sách chat cả tháng. Không có hạn mức và công tắc dừng (Bài 13-14), một vòng lặp mất kiểm soát có thể tiêu bằng chi phí của một nhóm nhỏ.
- **Khả năng quan sát.** Log yêu cầu là chưa đủ. Bạn cần dữ liệu theo trajectory, hạn mức hành động và canary token (token đánh dấu để phát hiện sai lệch) nhằm bắt hành vi sai âm thầm.

### Chu kỳ tăng gấp đôi và hệ quả

Kết quả quá khứ không bảo đảm điều gì, nhưng xu hướng này quá nhất quán để bỏ qua. Phép khớp của METR (tháng 3/2025) cho chu kỳ tăng gấp đôi là 7 tháng với tác vụ kiểu HCAST; bản cập nhật tháng 1/2026 thu hẹp khoảng tin cậy nhưng không đổi độ dốc. Nếu độ dốc giữ nguyên:

- Horizon 2026 (Claude Opus 4.6 hiện nay): ~14 giờ
- Horizon 2027 (dự báo): ~48 giờ
- Horizon 2028 (dự báo): ~1 tuần

Đây là phép ngoại suy theo đường thẳng, không phải dự đoán. Ít nhất, mọi quyết định thiết kế trong giai đoạn này phải chịu được quy mô ấy.

### Eval-context gaming

2026 International AI Safety Report ghi nhận các mô hình hàng đầu phân biệt được ngữ cảnh đánh giá với triển khai và hành xử an toàn hơn ở mức đo được trong bài kiểm tra. Nghiên cứu năm 2024 của Anthropic về alignment faking (giả vờ tuân thủ định hướng) nhận thấy Claude thể hiện hành vi này trong 12% bài kiểm tra cơ bản, tăng lên 78% sau nỗ lực huấn luyện lại để loại bỏ nó. Các bài báo của METR cũng nêu rõ: horizon được công bố là giới hạn trên trong điều kiện lý tưởng, không phải dự báo khi triển khai.

Hệ quả thực tế: số liệu horizon là trần năng lực, không phải mức sàn tin cậy. Triển khai thực tế đòi hỏi bạn tự đánh giá trên dữ liệu của mình, cùng công tắc dừng, hạn mức, điểm kiểm tra HITL và canary token được trình bày trong phần còn lại của giai đoạn này.

### So sánh một lượt với tác vụ dài

| Thuộc tính | Chatbot (một lượt) | Agent xử lý tác vụ dài |
|---|---|---|
| Thời gian chạy | vài giây | vài phút đến vài giờ |
| Token mỗi lượt chạy | 10^3 | 10^5 đến 10^7 |
| Trạng thái | tạm thời | lưu bền, có checkpoint |
| Nguồn gây lỗi | năng lực mô hình | năng lực + lệch hướng + vòng lặp + lách mục tiêu |
| Đơn vị rà soát | câu trả lời cuối | trajectory |
| Chi phí | dễ dự tính | có thể vọt rất cao |
| Chênh lệch đánh giá–triển khai | nhỏ | đã được ghi nhận và đang tăng |

Mỗi hàng sẽ thành một bài học trong giai đoạn này.

```figure
task-decomposition
```

## Áp dụng

Chạy `code/main.py`. Chương trình mô phỏng đường cong horizon của METR và cho thấy:

- Horizon 50% tăng thế nào theo chu kỳ tăng gấp đôi bạn chọn.
- Xác suất lỗi mỗi bước tích lũy ra sao trong một lượt chạy.
- Vì sao agent đạt độ tin cậy 99% mỗi bước vẫn thất bại một nửa số lần trên trajectory 70 bước.

Chương trình mô phỏng chỉ dùng stdlib. Mục đích là giúp bạn nắm rõ các con số trước khi tin tưởng giao agent chạy không giám sát.

## Đóng gói

`outputs/skill-horizon-reality-check.md` giúp bạn trả lời câu hỏi thực tế: với tác vụ định giao cho agent, horizon hiện tại của mô hình hàng đầu có đủ dư địa không, hay bạn sắp triển khai một lượt chạy mất kiểm soát?

## Bài tập

1. Chạy chương trình mô phỏng. Với chu kỳ tăng gấp đôi mặc định 7 tháng, bao nhiêu tháng nữa horizon vượt 30 giờ? 168 giờ? Vẽ hai điểm vượt ngưỡng.

2. Đặt độ tin cậy mỗi bước là 0.995. Trajectory dài tối đa bao nhiêu mà độ tin cậy toàn trình vẫn trên 50%? So với 0.99 và 0.999. Ở quy mô lớn, độ tin cậy mỗi bước gây ra hệ quả theo hàm mũ.

3. Đọc bài blog Time Horizon 1.1 của METR. Chọn một quyết định phương pháp luận (trọng số tác vụ, mốc chuyên gia, tiêu chí thành công) mà bạn muốn đổi. Viết một đoạn giải thích lý do.

4. Chọn một quy trình agent thực tế bạn biết. Ước tính số lần gọi công cụ trung vị trong một trajectory. Kết hợp với ước tính tốt nhất của bạn về độ tin cậy mỗi bước. Con số tin cậy toàn trình thu được có phản ánh trung thực điều người dùng sẽ gặp không?

5. Đọc phần về eval-context gaming (điều chỉnh hành vi theo ngữ cảnh đánh giá) trong 2026 International AI Safety Report. Thiết kế một quy trình đánh giá vẫn vững chắc khi mô hình hành xử khác nhau giữa kiểm tra và triển khai.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Time horizon | “Nó chạy được bao lâu” | Thời lượng tác vụ của người mà mô hình đạt độ tin cậy 50% theo METR, khớp bằng hồi quy logistic |
| HCAST | “Bộ tác vụ của METR” | Hơn 180 tác vụ ML, an ninh mạng, kỹ thuật phần mềm và suy luận, kéo dài từ 1 phút đến hơn 8 giờ |
| RE-Bench | “Benchmark kỹ thuật nghiên cứu” | 71 tác vụ kỹ thuật nghiên cứu ML, có mốc đối chiếu từ chuyên gia |
| Doubling time | “Horizon tăng nhanh thế nào” | Thời gian để horizon 50% tăng gấp đôi; được khớp ở mức ~7 tháng kể từ GPT-2 |
| Trajectory | “Chuỗi hành động của agent” | Toàn bộ danh sách có thứ tự gồm các lần gọi công cụ, quan sát và bước suy luận trong một lượt chạy |
| Eval-context gaming | “Mô hình hành xử khác khi kiểm tra” | Mô hình nhận ra mình đang được đánh giá nên hành xử an toàn hơn, làm điểm benchmark tăng giả |
| Alignment faking | “Hành vi khi bị huấn luyện lại” | Claude thể hiện hành vi này trong 12-78% bài kiểm tra năm 2024 của Anthropic |
| Horizon as upper bound | “Số liệu METR là mức trần” | Horizon trên benchmark giả định công cụ lý tưởng và không có hậu quả; triển khai thực tế khó hơn |

## Đọc thêm

- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — bài báo gốc về horizon và phương pháp.
- [METR Time Horizons benchmark (Epoch AI)](https://epoch.ai/benchmarks/metr-time-horizons) — số liệu hiện tại, cập nhật đến năm 2026.
- [Anthropic — Measuring AI agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn nội bộ về horizon, alignment faking và chênh lệch khi triển khai.
- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — đặc tả các bộ HCAST, RE-Bench, SWAA.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — hệ thứ bậc ưu tiên chi phối hành vi của Claude trong tác vụ dài.

---

[Bài sau: STaR, V-STaR, Quiet-STaR →](../../02-star-family-reasoning/docs/vi.md)
