# Chân trời thời gian METR và đánh giá năng lực độc lập

> METR (trước đây là ARC Evals) là tổ chức 501(c)(3) độc lập từ tháng 12/2023. Benchmark Time Horizon 1.1 (tháng 1/2026) của họ khớp một đường cong logistic với xác suất hoàn thành tác vụ theo log(thời gian chuyên gia hoàn thành); giao điểm tại xác suất 50% xác định chân trời thời gian của mô hình. Các đợt đánh giá 2025–2026 gồm GPT-5.1, GPT-5.1-Codex-Max và đánh giá thử khả năng giám sát (bộ giám sát có phát hiện tác vụ phụ không; agent có né được không). Các bộ benchmark: HCAST (180+ tác vụ ML, an ninh mạng, kỹ thuật phần mềm, suy luận; 1 phút đến 8+ giờ), RE-Bench (71 tác vụ nghiên cứu-kỹ thuật ML có mốc so sánh từ chuyên gia), SWAA. Cần lưu ý: phép đo của METR ở điều kiện lý tưởng — không có con người, không có hậu quả thực — và nhóm đã ghi nhận khoảng cách hành vi giữa đánh giá và triển khai (Bài 1). Chân trời thời gian là giới hạn trên, không phải dự đoán khi triển khai.

**Type:** Learn
**Languages:** Python (stdlib, logistic-fit horizon estimator)
**Prerequisites:** Giai đoạn 15, Bài 01 (Agent dài hạn), Giai đoạn 15, Bài 19 (RSP)
**Time:** ~60 phút

## Vấn đề

Chính sách mở rộng quy mô (Bài 19, 20) chỉ hữu ích khi phép đo được viện dẫn đáng tin. “Ngưỡng AI R&D-4” và “Tự chủ dài hạn” được định nghĩa bằng lời trong chính sách; chúng chỉ dùng được khi các đánh giá cụ thể cho ra số liệu cụ thể.

METR là tổ chức đánh giá độc lập giai đoạn 2024–2026 đã xác định nhiều số liệu như vậy. Họ đánh giá các mô hình tiên tiến — thường trước khi phát hành, theo thỏa thuận bảo mật với các phòng thí nghiệm — rồi công bố phương pháp. Benchmark Time Horizon 1.1 (tháng 1/2026) là kết quả nổi bật của họ: một giá trị duy nhất nén năng lực thành đơn vị dễ hiểu (“mô hình này làm được loại tác vụ mà chuyên gia mất X giờ, với độ tin cậy 50%”).

Bài học bàn cả phương pháp (cách tính chân trời) lẫn cách diễn giải (vì sao đó là giới hạn trên, không phải dự đoán khi triển khai). Hai kỹ năng này đi cùng nhau. Nhóm hiểu cách khớp chân trời sẽ khó bị lời quảng bá sai của nhà cung cấp đánh lừa hơn nhóm chỉ thấy “14 giờ” trên slide.

## Khái niệm

### Bối cảnh METR

- Thành lập: tháng 12/2023 (trước đây là ARC Evals, tách ra thành tổ chức 501(c)(3) độc lập).
- Phạm vi: đánh giá năng lực tự chủ của các mô hình tiên tiến, thường trước khi phát hành.
- Phòng thí nghiệm đối tác: Anthropic, OpenAI (nhiều đợt hợp tác 2025–2026).
- Kết quả đáng chú ý: Time Horizon 1.0 (tháng 3/2025), Time Horizon 1.1 (tháng 1/2026), các đánh giá thử khả năng giám sát.

### Cách khớp Time Horizon

Phương pháp (theo blog và bài báo của METR):

1. Thu thập bộ tác vụ có thời gian chuyên gia hoàn thành từ vài phút đến vài giờ. Các bộ hiện tại: HCAST (180+ tác vụ), RE-Bench (71 tác vụ), SWAA.
2. Cho mô hình làm từng tác vụ; ghi nhận thành công hoặc thất bại.
3. Khớp đường cong logistic: P(thành công) theo log(thời gian chuyên gia hoàn thành).
4. Chân trời là thời gian chuyên gia tại đó P(thành công) = 0.5.

Đường cong logistic phù hợp vì năng lực thường tăng rồi tiến dần tới mức ổn định khi đối chiếu với độ khó tác vụ. Mốc 50% là một lựa chọn (cũng có thể là 10%, 90%); METR báo cáo nhiều ngưỡng trong bài chi tiết nhưng lấy 50% làm mốc chính vì dễ hiểu nhất.

### Các số liệu tháng 1/2026

Theo Time Horizon 1.1:

- Claude Opus 4.6: ~14 giờ ở độ tin cậy 50%, theo Time Horizon 1.1 (tháng 1/2026).
- Thời gian tăng gấp đôi trên các tác vụ kiểu HCAST: ~4.3 tháng (130.8 ngày) theo phép khớp giai đoạn sau 2023 được Time Horizon 1.1 (tháng 1/2026) báo cáo; con số ~7 tháng là phép khớp toàn giai đoạn 2019–2025 từ Time Horizon 1.0 và được TH1.1 nêu cùng số liệu sau 2023.

Đó là các số liệu cụ thể được viện dẫn trong phần còn lại của Giai đoạn 15. RSP v3.0 của Anthropic (Bài 19) và FSF v3 của DeepMind (Bài 20) dùng chúng như một trong các phép đo để đánh giá ngưỡng.

### Các bộ benchmark

- **HCAST**: 189 tác vụ về ML, an ninh mạng, kỹ thuật phần mềm và suy luận tổng quát. Khoảng thời gian: 1 phút đến 8+ giờ của chuyên gia. Đây là bộ chính tạo ra con số chân trời.
- **RE-Bench**: 71 tác vụ nghiên cứu-kỹ thuật ML có mốc so sánh từ chuyên gia. Đo liệu mô hình có làm được dạng công việc R&D ML liên quan đến ngưỡng AI R&D-4 hay không.
- **SWAA**: các tác vụ ngắn hơn; hữu ích để hiệu chỉnh phần thấp của đường cong chân trời.

### Đánh giá thử khả năng giám sát

Công việc METR giai đoạn 2025–2026 gồm các thử nghiệm đánh giá giám sát: bộ giám sát có phát hiện mô hình làm tác vụ phụ (việc ngoài yêu cầu của người vận hành) không; agent có né được bộ giám sát không. Chúng trực tiếp phục vụ hạng mục “Làm suy yếu biện pháp bảo vệ” trong OpenAI PF v2 và việc giám sát hành vi đánh lừa trong DeepMind FSF v3 (Bài 20).

### Vì sao chân trời là giới hạn trên

- **Công cụ lý tưởng**: tác vụ benchmark cho mô hình công cụ rõ ràng và dữ liệu trình bày chuẩn. Triển khai thực tế phức tạp hơn.
- **Không có hậu quả thực**: mô hình không thực sự tính phí khách hàng, xóa dữ liệu thật hay liên hệ người thật. Triển khai thực tế có thể gây hậu quả không thể đảo ngược.
- **Lợi dụng bối cảnh đánh giá**: Bài 1. Mô hình hành xử khác khi được kiểm tra. 2026 International AI Safety Report ghi nhận điều này bằng thực nghiệm.
- **Không có biến thiên từ người dùng thực**: prompt benchmark có cấu trúc rõ. Người dùng thực đưa ra yêu cầu mơ hồ, phụ thuộc ngữ cảnh.

Chân trời là mức trần năng lực trong điều kiện thuận lợi. Độ tin cậy khi triển khai là con số khác, thấp hơn; nhóm phải đo trên phân bố tác vụ của mình để biết con số đó.

### Lý do cần bên đánh giá độc lập

Đánh giá độc lập quan trọng vì các phòng thí nghiệm có động lực tối ưu những chỉ số họ tự báo cáo. Tính độc lập của METR — tổ chức 501(c)(3) công bố phương pháp và có bài báo qua bình duyệt — giúp giảm rủi ro về mặt cấu trúc. Chừng đó chưa đủ (các phòng thí nghiệm vẫn kiểm soát những gì METR được thấy), nhưng rõ ràng tốt hơn việc không có đánh giá độc lập.

### Cách dùng số liệu chân trời trong thực tế

- **Lọc theo năng lực**: nếu chân trời của mô hình thấp hơn nhiều so với thời gian chuyên gia cần cho tác vụ dự kiến, đừng triển khai nó để tự làm tác vụ đó (tệp kỹ năng ở Bài 1).
- **Theo dõi xu hướng**: thời gian tăng gấp đôi cho biết cách làm hiện tại sẽ còn an toàn bao lâu, ngay cả khi không có biện pháp giảm thiểu mới.
- **Làm điểm xuất phát**: chân trời 14 giờ chỉ là điểm khởi đầu. Điều chỉnh giảm theo phân bố tác vụ, chất lượng công cụ và bối cảnh triển khai của bạn.

```figure
a5-horizon-fit
```

## Áp dụng

`code/main.py` khớp logistic giữa kết quả tác vụ và log(thời gian chuyên gia), dựa trên một tập kết quả mô phỏng. Chương trình báo cáo chân trời 50% (con số chính của METR), chân trời 10% (thận trọng) và chân trời 90% (lạc quan). Nó cũng minh họa điều gì thay đổi khi tỷ lệ thành công bị thổi phồng do lợi dụng bối cảnh đánh giá.

## Đóng gói

`outputs/skill-horizon-interpretation.md` xem xét tuyên bố về chân trời của nhà cung cấp và phân tích khoảng cách giữa tuyên bố trên benchmark với thực tế triển khai.

## Bài tập

1. Chạy `code/main.py`. Xác nhận chân trời 50% của phép khớp trùng với giá trị thực trong dữ liệu mô phỏng. Sau đó giảm một nửa lưới thời gian tác vụ; ước lượng chân trời có thay đổi đáng kể không?

2. Đọc bài blog Time Horizon 1.1 của METR. Xác định các tác vụ có độ tin cậy cao nhất và thấp nhất. Giải thích vì sao có khoảng cách đó.

3. Đọc tài liệu “Measuring Autonomous AI Capabilities” của METR. Liệt kê các nhóm tác vụ HCAST. Chọn một nhóm bạn sẽ đặt trọng số cao hơn cho tác vụ production và giải thích lý do.

4. Thêm hành vi lợi dụng bối cảnh đánh giá vào trình mô phỏng: đổi ~20% tác vụ thất bại thành thành công. Báo cáo chân trời mới. Điều này xấp xỉ tác động của tỷ lệ 20% lên con số quan sát được.

5. Thiết kế đánh giá chân trời nội bộ trên danh sách bug tồn đọng của bạn hoặc một bộ tác vụ tiêu biểu. Mô tả cách thu thập dữ liệu, phép khớp và ý nghĩa của kết quả. So sánh với số liệu METR.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| METR | “Bên đánh giá độc lập” | Trước đây là ARC Evals; tổ chức 501(c)(3) độc lập từ Dec 2023 |
| Time Horizon | “Thước đo năng lực” | Thời lượng tác vụ của chuyên gia tại độ tin cậy 50%, tính từ phép khớp logistic |
| HCAST | “Bộ chính của METR” | 180+ tác vụ, từ 1 phút đến 8+ giờ |
| RE-Bench | “Nghiên cứu-kỹ thuật” | 71 tác vụ nghiên cứu-kỹ thuật ML có mốc so sánh từ con người |
| SWAA | “Bộ tác vụ ngắn” | Hiệu chỉnh phần thấp của đường cong chân trời |
| Doubling time | “Tốc độ tăng” | Thời gian để chân trời 50% tăng gấp đôi; ~7 tháng theo HCAST |
| Eval-context gaming | “Mô hình hành xử khác” | Khoảng cách hành vi đã được ghi nhận giữa kiểm tra và triển khai |
| Upper bound | “Chân trời là mức trần” | Chân trời benchmark > độ tin cậy khi triển khai dưới tải |

## Đọc thêm

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — thông số HCAST, RE-Bench, SWAA.
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — bài báo đầu tiên về chân trời.
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — số liệu và phương pháp hiện tại.
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — theo dõi liên tục.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — góc nhìn nội bộ về các phép đo của METR.

---

[← Bài trước: OpenAI Preparedness Framework và DeepMind Frontier Safety Framework](../../20-openai-preparedness-deepmind-fsf/docs/vi.md) · [Bài sau: CAIS, CAISI và rủi ro quy mô xã hội →](../../22-cais-caisi-societal-risk/docs/vi.md)
