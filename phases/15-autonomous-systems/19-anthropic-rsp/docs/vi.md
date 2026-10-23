# Chính sách mở rộng có trách nhiệm v3.0 của Anthropic

> RSP v3.0 có hiệu lực từ ngày 24/2/2026, thay chính sách 2023. Biện pháp giảm thiểu chia hai tầng: việc Anthropic tự thực hiện và khuyến nghị cho toàn ngành (gồm chuẩn bảo mật RAND SL-4). Frontier Safety Roadmaps và Risk Reports trở thành tài liệu thường trực thay vì sản phẩm làm một lần. Cam kết tạm dừng năm 2023 bị bỏ. Chính sách đưa ra ngưỡng AI R&D-4: khi vượt ngưỡng, Anthropic phải công bố lập luận xác nhận, nêu rủi ro sai lệch mục tiêu và biện pháp giảm thiểu. Claude Opus 4.6 chưa vượt ngưỡng. Trong thông báo v3.0, Anthropic nói rằng "việc tự tin loại trừ khả năng này ngày càng khó." SaferAI chấm RSP 2023 ở mức 2.2; họ hạ v3.0 xuống 1.9, xếp Anthropic vào nhóm RSP "yếu" cùng OpenAI và DeepMind. Các ngưỡng định tính thay cam kết định lượng năm 2023; bỏ điều khoản tạm dừng là bước lùi rõ nhất.

**Type:** Learn
**Languages:** Python (stdlib, RSP threshold decision engine)
**Prerequisites:** Giai đoạn 15, Bài 06 (AAR), Giai đoạn 15, Bài 07 (RSI)
**Time:** ~45 phút

## Vấn đề

Các phòng thí nghiệm mô hình tiên tiến công bố chính sách mở rộng: vừa là tài liệu kỹ thuật, vừa là tài liệu quản trị, vừa là tín hiệu gửi cơ quan quản lý. RSP v3.0 là tài liệu hiện hành của Anthropic. Cần đọc kỹ không phải vì Anthropic buộc phải tuân thủ (họ không bị ràng buộc), mà vì cách trình bày định hình cách họ nhìn nhận rủi ro thảm khốc và truyền đạt những đánh đổi với công chúng.

Điều hữu ích là so sánh v3.0 với v2.0. Phần được thêm: Frontier Safety Roadmaps, Risk Reports, ngưỡng AI R&D-4. Phần bị bỏ: cam kết tạm dừng năm 2023. Phần đổi cách trình bày: lịch trình giảm thiểu hai tầng, tách việc Anthropic tự làm khỏi khuyến nghị cho ngành. Đánh giá bên ngoài — SaferAI — hạ điểm từ 2.2 (v2) xuống 1.9 (v3.0). Một chính sách mở rộng có thể kém chặt chẽ hơn dù trông chỉn chu hơn.

## Khái niệm

### Lịch trình giảm thiểu hai tầng

- **Việc Anthropic tự thực hiện**: những gì Anthropic sẽ làm bất kể các phòng thí nghiệm khác làm gì. Dừng huấn luyện khi vượt ngưỡng, áp dụng biện pháp bảo mật cụ thể, đặt điều kiện triển khai cụ thể.
- **Khuyến nghị cho toàn ngành**: những gì Anthropic cho rằng cả ngành nên cùng làm. Gồm chuẩn bảo mật RAND SL-4. Đây không phải cam kết của Anthropic mà là đề xuất chính sách.

Cấu trúc hai tầng không có trong v2. Vì vậy, bạn cần xem mỗi cam kết nằm ở cột nào. Biện pháp bảo mật trong cột "khuyến nghị cho toàn ngành" không phải lời hứa của Anthropic; đó là điều họ mong muốn.

### Ngưỡng AI R&D-4

Đây là mức năng lực mà RSP v3.0 xác định là ngưỡng quan trọng tiếp theo. Cụ thể: một mô hình có thể tự động hóa phần đáng kể của nghiên cứu AI với chi phí cạnh tranh. Khi Anthropic tin rằng mô hình vượt ngưỡng, họ phải công bố lập luận xác nhận, nêu rủi ro sai lệch mục tiêu và biện pháp giảm thiểu trước khi tiếp tục mở rộng.

Theo thông báo v3.0, Claude Opus 4.6 chưa vượt ngưỡng. Tài liệu nói thêm: "việc tự tin loại trừ khả năng này ngày càng khó." Cách nói ấy quan trọng: nó thừa nhận ngưỡng đã đủ gần để cần quan tâm ngay, không phải giới hạn giả định xa xôi.

Bài 6 (Nghiên cứu căn chỉnh tự động) và Bài 7 (Tự cải thiện đệ quy) liên quan trực tiếp đến ngưỡng này. Việc các nhà nghiên cứu căn chỉnh tự động đạt chuẩn chất lượng nghiên cứu là bằng chứng ngưỡng AI R&D-4 đang đến gần.

### Frontier Safety Roadmaps và Risk Reports

v3.0 đưa hai loại sản phẩm thành tài liệu thường trực:

- **Frontier Safety Roadmap**: tài liệu hướng tới tương lai, mô tả công việc an toàn dự kiến, năng lực kỳ vọng và nghiên cứu giảm thiểu.
- **Risk Report**: tài liệu nhìn lại từng mô hình sau khi phát hành, mô tả năng lực quan sát được và rủi ro còn lại.

Cả hai đều công khai và được cập nhật theo lịch đã công bố. Nhờ đó, bạn có thể đối chiếu việc Anthropic nói sẽ làm trong Roadmap với những gì họ báo cáo trong Risk Report.

### Bỏ điều khoản tạm dừng

RSP 2023 có cam kết tạm dừng rõ ràng: nếu mô hình vượt các ngưỡng năng lực cụ thể, việc huấn luyện sẽ dừng đến khi có biện pháp giảm thiểu. v3.0 thay cam kết đó bằng quy định mềm hơn (công bố lập luận xác nhận, tiếp tục nếu biện pháp giảm thiểu phù hợp). SaferAI và các nhà phân tích khác chỉ rõ đây là bước lùi lớn nhất của tài liệu mới.

Lý do chính sách cho thay đổi này: đến năm 2026, các ngưỡng định lượng năm 2023 không còn áp dụng được vì chính các thước đo năng lực đã được điều chỉnh lại. Lập luận phản bác: điều khoản tạm dừng trong chính sách mở rộng là cơ chế ràng buộc cam kết; bỏ nó làm chính sách mất độ tin cậy.

### SaferAI hạ điểm

SaferAI là tổ chức độc lập chấm các tài liệu dạng RSP. Điểm công khai của họ: RSP Anthropic 2023 đạt 2.2 (trên thang mà 4.0 là RSP hiện tốt nhất và 1.0 là mức danh nghĩa). v3.0 đạt 1.9. Anthropic vì thế chuyển từ nhóm "trung bình" sang "yếu", cùng nhóm với OpenAI và DeepMind.

Các yếu tố khiến SaferAI hạ điểm:
- Ngưỡng định tính thay ngưỡng định lượng.
- Cam kết tạm dừng bị bỏ.
- Biện pháp giảm thiểu cho ngưỡng AI R&D-4 được mô tả là "lập luận xác nhận" thay vì các biện pháp cụ thể.
- Cơ chế rà soát phụ thuộc vào Safety Advisory Group của Anthropic, với ít giám sát độc lập.

### Bài này không nói về điều gì

Đây không phải bài học về tuân thủ. RSP v3.0 không phải quy định pháp luật; không gì buộc Anthropic làm theo. Bài học là đọc tài liệu đủ cụ thể và thận trọng. Chính sách mở rộng là tín hiệu công khai chính mà các phòng thí nghiệm mô hình tiên tiến đưa ra về cách ứng phó rủi ro thảm khốc. Đọc tốt chúng là kỹ năng thực tế cho bất kỳ ai có công việc phụ thuộc vào năng lực mô hình tiên tiến.

```figure
a5-rsp-ladder
```

## Áp dụng

`code/main.py` triển khai bộ ra quyết định nhỏ theo cách RSP đánh giá ngưỡng: nhận mô hình ứng viên và các phép đo năng lực, rồi trả về liệu mô hình có vượt ngưỡng AI R&D-4 không, các mục bắt buộc của lập luận xác nhận và liệu có thể triển khai không. Nó cố ý đơn giản; mục đích là làm rõ logic của tài liệu.

## Đóng gói

`outputs/skill-scaling-policy-review.md` rà soát một chính sách mở rộng (của Anthropic, OpenAI, DeepMind hoặc nội bộ) theo bản tham chiếu v3.0: cấu trúc hai tầng, ngưỡng, cam kết tạm dừng, rà soát độc lập.

## Bài tập

1. Chạy `code/main.py`. Nhập ba mô hình giả lập ở các mức năng lực khác nhau. Xác nhận bộ đánh giá ngưỡng hoạt động đúng dự kiến và tạo đúng mẫu lập luận xác nhận.

2. Đọc toàn bộ RSP v3.0 (32 trang). Tìm mọi cam kết thuộc tầng "khuyến nghị cho toàn ngành". Cam kết nào trong số đó sẽ thuộc tầng "Anthropic tự thực hiện" ở v2?

3. Đọc phương pháp chấm RSP của SaferAI. Áp dụng thang chấm của họ cho tài liệu để tái tạo điểm 1.9 của v3.0. Mục nào trong thang chấm khiến điểm giảm nhiều nhất?

4. Cam kết tạm dừng năm 2023 đã bị bỏ. Hãy đề xuất cam kết thay thế, vừa giữ độ tin cậy của chính sách vừa thừa nhận vấn đề điều chỉnh lại thước đo năm 2026.

5. So sánh RSP v3.0 với OpenAI Preparedness Framework v2 (Bài 20). Chọn một điểm v3.0 mạnh hơn và một điểm Preparedness Framework mạnh hơn.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| RSP | "Chính sách mở rộng của Anthropic" | Chính sách mở rộng có trách nhiệm; v3.0 có hiệu lực Feb 24, 2026 |
| AI R&D-4 | "Ngưỡng tự động hóa nghiên cứu" | Năng lực tự động hóa phần đáng kể của nghiên cứu AI với chi phí cạnh tranh |
| Affirmative case | "Lý giải về an toàn" | Lập luận công khai rằng rủi ro đã được xác định và biện pháp giảm thiểu phù hợp |
| Frontier Safety Roadmap | "Kế hoạch sắp tới" | Tài liệu thường trực về công việc an toàn dự kiến và năng lực kỳ vọng |
| Risk Report | "Đánh giá lại một mô hình" | Tài liệu thường trực về năng lực quan sát được và rủi ro còn lại sau phát hành |
| Two-tier mitigation | "Tự thực hiện và toàn ngành" | Tách cam kết của Anthropic khỏi khuyến nghị cho ngành |
| Pause commitment | "Điều khoản 2023" | Lời hứa rõ ràng sẽ tạm dừng huấn luyện; bị bỏ trong v3.0 |
| SaferAI rating | "Điểm RSP độc lập" | Thang chấm của bên thứ ba; v3.0 đạt 1.9 (v2 đạt 2.2) |

## Đọc thêm

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — toàn văn chính sách 32 trang.
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — tóm tắt thay đổi so với v2.
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/responsible-scaling-policy/roadmap) — tài liệu thường trực được dẫn từ RSP v3.0.
- [Anthropic — Risk Report: February 2026](https://www.anthropic.com/feb-2026-risk-report) — đánh giá lại Claude Opus 4.6, mô hình tiên tiến tại thời điểm công bố.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — liên hệ AI R&D-4 với mức tự chủ đo được.

---

[← Bài trước: Llama Guard và phân loại đầu vào/đầu ra](../../18-llama-guard/docs/vi.md) · [Bài sau: OpenAI Preparedness Framework và DeepMind Frontier Safety Framework →](../../20-openai-preparedness-deepmind-fsf/docs/vi.md)
