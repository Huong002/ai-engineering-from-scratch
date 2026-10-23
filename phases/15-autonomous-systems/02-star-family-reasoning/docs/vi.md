# STaR, V-STaR, Quiet-STaR — Suy luận tự học

> Vòng lặp tự cải thiện nhỏ nhất nằm trong lời giải. Mô hình tạo chuỗi suy nghĩ, giữ những chuỗi dẫn đến đáp án đúng rồi fine-tune trên chúng. Đó là STaR. V-STaR thêm bộ kiểm chứng để chọn tốt hơn lúc suy luận. Quiet-STaR đưa lời giải xuống từng token. Cả ba đều hiệu quả. Nhưng không có phép màu nào: vòng lặp giữ lại cả lối tắt tình cờ dẫn đến đáp án đúng.

**Type:** Learn
**Languages:** Python (stdlib, bootstrap-loop simulator)
**Prerequisites:** Giai đoạn 13, Bài 01-03 (Suy luận và CoT), Giai đoạn 15, Bài 01 (khung tác vụ dài hạn)
**Time:** ~60 phút

## Vấn đề

Cách trực tiếp để dạy mô hình suy luận là thu thập lời giải do người viết. Việc này tốn kém, chậm và bị giới hạn bởi lượng chuỗi suy nghĩ chất lượng cao mà con người sẵn lòng viết.

STaR (Self-Taught Reasoner, mô hình suy luận tự học; Zelikman et al., 2022) đặt câu hỏi: nếu mô hình tự viết lời giải rồi đối chiếu với đáp án đã biết thì sao? Vòng lặp là:

1. Lấy mẫu một lời giải kèm đáp án.
2. Nếu đáp án cuối đúng, giữ lời giải.
3. Fine-tune trên các lời giải được giữ.
4. Lặp lại.

Cách này hiệu quả. Kết quả trên GSM8K và CommonsenseQA đều cải thiện mà không cần gán nhãn mới bằng tay. Nhưng vòng lặp có thiên lệch sẵn có: lời giải nào cho đáp án đúng đều được giữ, dù lập luận có vững hay không. V-STaR (Hosseini et al., 2024) khắc phục bằng bộ kiểm chứng học được; Quiet-STaR (Zelikman et al., 2024) mở rộng ý tưởng sang lời giải nội tại cho từng token.

## Khái niệm

### STaR: học tiếp từ những gì hiệu quả

Bắt đầu với mô hình cơ sở có chút khả năng suy luận. Với mỗi bài toán huấn luyện, lấy mẫu lời giải kèm đáp án. Nếu đáp án khớp nhãn, giữ bộ ba (bài toán, lời giải, đáp án). Fine-tune mô hình trên tập được giữ. Lặp lại.

Có một điểm quan trọng. Nếu mô hình không bao giờ giải đúng một bài, vòng lặp không thể học từ bài đó. STaR thêm **rationalization (tạo lời giải theo đáp án)**: với bài mô hình làm sai, đưa đáp án đúng làm gợi ý rồi yêu cầu mô hình tạo lời giải dẫn đến đáp án ấy. Các lời giải được tạo theo cách này được thêm vào tập huấn luyện.

Kết quả trong bài báo gốc (Zelikman et al., 2022): mô hình GPT-J cơ sở tăng từ 5.8% lên 10.7% trên GSM8K sau nhiều vòng STaR có gợi ý đáp án — tăng tuyệt đối khoảng 5 điểm phần trăm. Trên CommonsenseQA, GPT-J 6B huấn luyện bằng STaR đạt 72.5%, tương đương GPT-3 175B đã fine-tune (~73%) — mô hình lớn hơn khoảng 30x, được huấn luyện trên lời giải gán nhãn bằng tay.

### V-STaR: huấn luyện bộ kiểm chứng bằng DPO

STaR bỏ các lời giải sai. Hosseini et al. (2024) nhận thấy chúng cũng là dữ liệu: mỗi cặp (lời giải, "có đúng không") đều có thể dùng để huấn luyện bộ kiểm chứng. Họ dùng Direct Preference Optimization trên cả lời giải đúng lẫn sai để xây dựng bộ xếp hạng. Lúc suy luận, lấy mẫu N lời giải và chọn phương án bộ kiểm chứng xếp cao nhất.

Mức tăng được báo cáo: thêm +4 đến +17 điểm phần trăm so với các phương pháp tự cải thiện trước đó trên GSM8K và MATH. Phần lớn lợi ích đến từ việc dùng bộ kiểm chứng để chọn lúc suy luận, thay vì fine-tune thêm mô hình sinh.

### Quiet-STaR: lời giải nội tại cho từng token

Zelikman et al. (2024) đặt câu hỏi: nếu mô hình học tạo một lời giải nội tại ngắn ở mỗi vị trí token, thay vì chỉ giữa bài toán và đáp án, thì sao? Quiet-STaR huấn luyện mô hình tạo một "suy nghĩ" ẩn trước mỗi token cần dự đoán, rồi kết hợp dự đoán có suy nghĩ với dự đoán cơ sở bằng trọng số học được.

Kết quả: Mistral 7B tăng điểm zero-shot tuyệt đối trên GSM8K từ 5.9% lên 10.9% và trên CommonsenseQA từ 36.3% lên 47.2% mà không cần fine-tune riêng cho tác vụ. Mô hình học được "khi nào cần nghĩ": token khó có lời giải nội tại dài hơn; token dễ hầu như không có.

### Vì sao cả ba cùng có rủi ro an toàn

Cả ba phương pháp đều dùng đáp án cuối làm tín hiệu gradient. Lời giải đi đến đáp án đúng bằng lập luận sai — khai thác lối tắt, đoán mò hoặc dùng mẫu không khái quát được — vẫn được củng cố. Với bài toán cùng phân phối, lối tắt có tác dụng. Với bài toán ngoài phân phối, nó âm thầm thất bại.

Bộ kiểm chứng của V-STaR giảm rủi ro bằng cách học xếp hạng lời giải, nhưng nó được huấn luyện trên cùng tập nhãn. Nó có thể chuộng lập luận sai trình bày đẹp hơn sự thừa nhận không chắc chắn. Thiết kế an toàn hơn là kết hợp dữ liệu kiểu STaR với (a) mô hình thưởng giám sát quá trình, thưởng các bước trung gian chứ không chỉ đáp án, và (b) đánh giá ngoài phân phối (OOD) trên tập giữ riêng nhằm phá các lối tắt đơn giản.

### So sánh

| Phương pháp | Tín hiệu huấn luyện | Chi phí suy luận | Dữ liệu bị bỏ | Lỗi đã biết |
|---|---|---|---|---|
| STaR | giữ (lời giải, đáp án) nếu đúng | 1x | bỏ mọi lời giải sai | lời giải dùng lối tắt |
| STaR + gợi ý đáp án | như trên + thử lại với đáp án đúng được gợi ý | 1x | ít hơn | lời giải tạo theo đáp án có thể thiếu hợp lý |
| V-STaR | STaR + bộ kiểm chứng DPO từ cả hai nhóm | Nx (chọn tốt nhất trong N) | tối thiểu | bộ kiểm chứng có thể củng cố sai lầm đầy tự tin |
| Quiet-STaR | lời giải từng token + trọng số kết hợp | 1.5-3x | tối thiểu | gradient vẫn phụ thuộc đáp án |

### Vị trí trong hệ công cụ năm 2026

STaR đã ra đời từ lâu. Nhưng mô thức này xuất hiện khắp nơi trong 2025-2026. RL trên bài toán có đáp án kiểm chứng được (DeepSeek-R1, Kimi-k1.5, o1) dùng tín hiệu gradient phụ thuộc đáp án của STaR ở quy mô lớn hơn. Mô hình thưởng quá trình (Lightman et al., 2023; "Let's verify step by step" của OpenAI) là lựa chọn giám sát quá trình. AlphaEvolve (Bài 3) là STaR cho code, với bộ đánh giá chương trình thay cho nhãn. Darwin Godel Machine (Bài 4) là STaR cho chính cấu trúc hỗ trợ agent.

Hiểu STaR sẽ giúp bạn hiểu tất cả những cách này. Đây là vòng lặp tự cải thiện tối thiểu khả dụng.

```figure
reflection-loop
```

## Áp dụng

`code/main.py` chạy vòng lặp STaR mô phỏng trên bài toán số học đơn giản. Bạn có thể quan sát:

- Độ chính xác tăng qua các vòng học từ kết quả trước.
- Lối tắt len vào thế nào: mô phỏng có một loại lời giải "lười" cho đáp án đúng 40% số lần nhưng khái quát kém. Hãy xem STaR có giữ chúng không.
- Bộ kiểm chứng (theo kiểu V-STaR) giúp ích lúc suy luận ra sao nhưng không thể loại hết lối tắt đã lọt vào khi huấn luyện.

## Đóng gói

`outputs/skill-star-loop-reviewer.md` giúp bạn kiểm tra một pipeline suy luận tự học được đề xuất trước khi huấn luyện trên đó.

## Bài tập

1. Chạy mô phỏng. Đặt tần suất lối tắt về 0, rồi 0.4. Độ chính xác cuối chênh bao nhiêu giữa hai lần chạy, dù cả hai đều đạt >90% trên phân phối huấn luyện?

2. Thêm bài kiểm tra OOD giữ riêng vào mô phỏng. Lấy bài toán từ phân phối khác và đánh giá mô hình sau các vòng học trên cả tập cùng phân phối lẫn OOD. Định lượng chênh lệch.

3. Đọc Section 3 của bài báo Quiet-STaR (arXiv:2403.09629). Giải thích token "end-of-thought" và đầu dự đoán trọng số kết hợp, mỗi phần trong ba câu.

4. So sánh bộ lọc giữ-nếu-đúng của STaR với phương án giám sát quá trình, thưởng độc lập từng bước lời giải. Nêu chênh lệch chi phí gán nhãn và chênh lệch chất lượng có thể có.

5. Thiết kế một phép đánh giá phát hiện lời giải dùng lối tắt trong mô hình đã triển khai. Không cần hoàn hảo — nó cần phá được những lối tắt đơn giản nhất mà vòng lặp STaR sẽ củng cố.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| STaR | "Self-Taught Reasoner" | Lặp lại việc fine-tune trên lời giải do mô hình tạo ra và dẫn đến đáp án đúng |
| Rationalization | "Thử lại với gợi ý" | Đưa đáp án đúng vào rồi yêu cầu tạo lại lời giải cho bài mà mô hình cơ sở làm sai |
| V-STaR | "STaR có bộ kiểm chứng" | Huấn luyện bộ kiểm chứng bằng DPO trên cả lời giải đúng và sai, rồi dùng nó để chọn lúc suy luận |
| Quiet-STaR | "Lời giải từng token" | Tạo suy nghĩ ẩn ở mỗi vị trí token; kết hợp với dự đoán cơ sở |
| Answer-conditioned gradient | "Tín hiệu dựa trên kết quả" | Vòng lặp huấn luyện thưởng đáp án cuối, không thưởng các bước suy luận |
| Process reward model | "Bộ kiểm chứng từng bước" | Mô hình thưởng được huấn luyện theo độ đúng của từng bước, không theo kết quả — khác STaR |
| Shortcut rationale | "Đáp án đúng, lập luận sai" | Lời giải đến được nhãn nhờ mẫu không khái quát được; STaR vẫn giữ lại |

## Đọc thêm

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — bài báo gốc.
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — thêm bộ kiểm chứng DPO để chọn lúc suy luận.
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — lời giải nội tại cho từng token.
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — mô hình thưởng quá trình, tín hiệu gradient thay thế.
- [Bài báo DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — RL trên tác vụ kiểm chứng được, mở rộng STaR lên huấn luyện mô hình hàng đầu.

---

[← Bài trước: Từ chatbot đến agent xử lý tác vụ dài](../../01-long-horizon-agents/docs/vi.md) · [Bài sau: AlphaEvolve →](../../03-alphaevolve-evolutionary-coding/docs/vi.md)
