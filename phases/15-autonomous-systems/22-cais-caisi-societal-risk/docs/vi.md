# CAIS, CAISI và rủi ro quy mô xã hội

> Center for AI Safety (CAIS, San Francisco, do Hendrycks và Zhang thành lập năm 2022) công bố khung bốn rủi ro — sử dụng ác ý, cuộc đua AI, rủi ro tổ chức, AI vượt kiểm soát — và tuyên bố tháng 5 năm 2023 về nguy cơ tuyệt chủng, được hàng trăm giáo sư và lãnh đạo công ty ký tên. Các ấn phẩm năm 2026 của CAIS gồm AI Dashboard để đánh giá mô hình tiên tiến, Remote Labor Index (cùng Scale AI), Superintelligence Strategy Paper và bản tin AI Frontiers. Một tổ chức khác là NIST Center for AI Standards and Innovation (CAISI): đơn vị thuộc chính phủ Mỹ, phụ trách thỏa thuận tự nguyện và đánh giá năng lực không mật, tập trung vào rủi ro mạng, sinh học và vũ khí hóa học. CAIS xếp rủi ro tổ chức vào một trong bốn nhóm rủi ro chính: văn hóa an toàn, kiểm toán chặt chẽ, phòng vệ nhiều lớp và an ninh thông tin là nền tảng, nhưng thường bị đánh đổi để triển khai nhanh hơn. California SB-53, nếu được ký thành luật, sẽ là quy định đầu tiên ở cấp bang của Mỹ về rủi ro thảm họa.

**Type:** Learn
**Languages:** Python (stdlib, four-risk inventory and mitigation matcher)
**Prerequisites:** Giai đoạn 15, Bài 19 (RSP), Giai đoạn 15, Bài 20 (PF + FSF)
**Time:** ~45 phút

## Vấn đề

Bài 19 và 20 đề cập các chính sách mở rộng quy mô nội bộ phòng thí nghiệm. Bài 21 đề cập đánh giá năng lực độc lập. Bài này xét góc nhìn thứ ba: các tổ chức xã hội dân sự và chính phủ định hình thảo luận công khai cùng nền tảng quy định về rủi ro AI thảm họa.

Cần phân biệt hai tổ chức. CAIS là tổ chức nghiên cứu phi lợi nhuận, công bố các khung tư duy về rủi ro AI và điều phối tuyên bố công khai. CAISI là trung tâm thuộc chính phủ Mỹ trong NIST, thực hiện thỏa thuận tự nguyện với các phòng thí nghiệm và đánh giá năng lực không mật. Tên viết tắt gần giống nhau; nhiệm vụ thì không trùng. Người làm trong lĩnh vực này nên biết cả hai.

Nội dung thực tiễn: khung bốn rủi ro của CAIS là cách phân loại rủi ro quy mô xã hội được trích dẫn nhiều nhất trong tài liệu. Văn hóa an toàn và rủi ro tổ chức thuộc một trong bốn nhóm đó, cũng là nhóm người làm trực tiếp kiểm soát được nhiều nhất. SB-53 (California) sẽ là quy định đầu tiên ở cấp bang của Mỹ về rủi ro thảm họa nếu được ký thành luật; cách dự luật đặt vấn đề rất quan trọng vì trong chính sách công nghệ Mỹ, quy định cấp bang thường đi trước hành động liên bang.

## Khái niệm

### CAIS — Center for AI Safety

- Thành lập: năm 2022 tại San Francisco, bởi Dan Hendrycks và đồng nghiệp (tên “Zhang” chỉ một cộng tác viên thời kỳ đầu, không phải đồng sáng lập hiện nay; xem trang CAIS để biết lãnh đạo hiện tại).
- Tư cách: tổ chức phi lợi nhuận 501(c)(3).
- Thành quả đáng chú ý năm 2023: tuyên bố về nguy cơ tuyệt chủng, được hàng trăm nhà nghiên cứu và CEO cùng ký. Tuyên bố viết: “Giảm thiểu nguy cơ tuyệt chủng do AI phải là ưu tiên toàn cầu, ngang với các rủi ro quy mô xã hội khác như đại dịch và chiến tranh hạt nhân.”
- Các ấn phẩm năm 2026: AI Dashboard để đánh giá mô hình tiên tiến, Remote Labor Index (cùng Scale AI), Superintelligence Strategy Paper và bản tin AI Frontiers.

### Khung bốn rủi ro

Khung của CAIS chia rủi ro AI thảm họa thành bốn nhóm chính:

1. **Sử dụng ác ý**: tác nhân xấu dùng AI để gây hại (tổng hợp vũ khí sinh học, tung tin sai lệch, tấn công mạng).
2. **Cuộc đua AI**: áp lực cạnh tranh giữa các phòng thí nghiệm, công ty hoặc quốc gia thúc đẩy triển khai vượt quá ngưỡng an toàn.
3. **Rủi ro tổ chức**: hoạt động nội bộ phòng thí nghiệm (văn hóa an toàn yếu, kiểm toán thiếu sót, an ninh thiếu nguồn lực) dẫn đến triển khai gây hại.
4. **AI vượt kiểm soát**: AI đủ năng lực theo đuổi mục tiêu xung đột với phúc lợi con người.

Đây không phải cách phân loại duy nhất, nhưng được trích dẫn nhiều nhất. Các nhóm không loại trừ nhau — một AI vượt kiểm soát do tổ chức đánh đổi kiểm toán lấy tốc độ trong cuộc đua tạo ra thuộc cả bốn nhóm.

### Rủi ro tổ chức nằm ở đâu

Trong bốn nhóm, rủi ro tổ chức là nhóm người làm trực tiếp có thể tác động nhiều nhất. Văn hóa an toàn, mức độ chặt chẽ của kiểm toán, các lớp phòng vệ và an ninh thông tin của phòng thí nghiệm quyết định liệu mô hình được phát hành với các biện pháp kiểm soát ở Bài 10–18 thực sự hoạt động, hay chúng chỉ là ô đánh dấu chưa ai kiểm chứng.

Các đòn bẩy cụ thể đối với rủi ro tổ chức:

- **Văn hóa an toàn**: thành viên nhóm có thể nêu quan ngại mà không sợ ảnh hưởng sự nghiệp không? Khảo sát của CAIS cho thấy đây là yếu tố dự báo mạnh các đòn bẩy còn lại.
- **Kiểm toán chặt chẽ**: cả bên ngoài lẫn nội bộ. Chỉ kiểm toán nội bộ thường cho ra báo cáo quá lạc quan.
- **Phòng vệ nhiều lớp**: không lớp nào đủ sức bảo vệ một mình (chủ đề xuyên suốt Giai đoạn 15).
- **An ninh thông tin**: rò rỉ trọng số mô hình, dữ liệu đánh giá hoặc kỹ thuật vượt qua hệ thống giám sát. RAND SL-4 trong Bài 19 là một tiêu chuẩn cụ thể.

### CAISI — Center for AI Standards and Innovation

- Hoạt động trong NIST.
- Thực hiện thỏa thuận tự nguyện với các phòng thí nghiệm phát triển mô hình tiên tiến.
- Công bố đánh giá năng lực không mật, tập trung vào rủi ro mạng, sinh học và vũ khí hóa học.
- Khác CAIS; hai tên viết tắt dễ nhầm; hãy kiểm tra URL (nist.gov) để biết bạn đang đọc về tổ chức nào.

CAISI là đối trọng công khai, thuộc chính phủ, với hoạt động hợp tác riêng của METR cùng các phòng thí nghiệm (Bài 21). Báo cáo CAISI không mật; báo cáo METR thường chỉ truy cập được theo NDA. Đọc cả hai giúp bạn có cái nhìn đầy đủ hơn.

### California SB-53

Dự luật của Thượng viện California (kỳ họp 2025–2026) đề cập rủi ro thảm họa từ mô hình tiên tiến. Các điều khoản chính trong bản dự thảo:

- Ngưỡng năng lực cụ thể làm phát sinh nghĩa vụ ở cấp bang.
- Bảo vệ người tố giác là nhân viên phòng thí nghiệm AI.
- Yêu cầu báo cáo sự cố thảm họa.

Nếu được ký thành luật, đây sẽ là quy định đầu tiên ở cấp bang của Mỹ về rủi ro thảm họa. Dù có được ký hay không, cách dự luật đặt vấn đề vẫn định hình cách các cơ quan lập pháp bang khác tiếp cận vấn đề. Người làm trong lĩnh vực này ở California nên theo dõi tình trạng dự luật; người ở nơi khác nên đọc để hiểu quy định cấp bang của Mỹ có thể sẽ ra sao.

### Rủi ro quy mô xã hội không thể giải quyết bằng một lớp

Chủ đề xuyên suốt Giai đoạn 15 — phòng vệ theo chiều sâu — cũng áp dụng ở cấp xã hội. Không tổ chức, quy định hay khung nào một mình loại bỏ được rủi ro thảm họa. Cả hệ thống chỉ hoạt động khi:

- Các phòng thí nghiệm áp dụng chính sách mở rộng quy mô (Bài 19, 20).
- Bên đánh giá độc lập cung cấp phép đo (Bài 21).
- Xã hội dân sự theo dõi và công bố thông tin (CAIS).
- Chính phủ vận hành chương trình tự nguyện và đặt nền tảng quy định (CAISI, SB-53).
- Người làm trực tiếp xây dựng biện pháp kiểm soát nhiều lớp (Bài 10–18).

Đây là phần tổng hợp cuối giai đoạn: mỗi bài trước là một lớp trong hệ thống, mà độ đầy đủ của cả hệ thống quan trọng hơn sức mạnh của bất kỳ lớp riêng lẻ nào.

```figure
a5-four-risks
```

## Áp dụng

`code/main.py` triển khai công cụ lập danh mục rủi ro nhỏ. Với một phương án triển khai, công cụ gắn nhãn theo bốn nhóm rủi ro và trả về danh sách biện pháp giảm thiểu cần kiểm tra. Nó giúp đọc hiểu khung, không thay thế phán đoán của con người.

## Đóng gói

`outputs/skill-societal-risk-review.md` xem xét mức độ sẵn sàng trước rủi ro quy mô xã hội của một phương án triển khai: phương án liên quan nhóm nào trong bốn nhóm, đã có biện pháp giảm thiểu nào và mức độ phơi nhiễm với rủi ro tổ chức ra sao.

## Bài tập

1. Chạy `code/main.py`. Nhập ba phương án triển khai giả lập ở các quy mô khác nhau. Xác nhận nhãn bốn rủi ro đúng như bạn dự kiến; tìm một trường hợp công cụ gắn nhãn thiếu hoặc thừa.

2. Đọc toàn bộ bài viết của CAIS về bốn rủi ro. Chọn một nhóm rủi ro và viết hai đoạn về diễn biến quan trọng nhất năm 2026 trong nhóm đó theo bạn.

3. Đọc bản dự thảo hiện hành của California SB-53. Tìm một điều khoản mà bạn cho là tăng khả năng ứng phó rủi ro thảm họa và một điều khoản làm giảm khả năng đó. Giải thích cả hai.

4. Chọn một hệ thống AI đang vận hành mà bạn biết (của bạn hoặc đã công bố). Đánh giá theo các đòn bẩy phụ của rủi ro tổ chức: văn hóa an toàn, mức độ chặt chẽ của kiểm toán, phòng vệ nhiều lớp, an ninh thông tin. Mục nào yếu nhất? Cần bao nhiêu nguồn lực để đưa nó lên ngang mức các mục khác?

5. Phác thảo phiên bản khung bốn rủi ro cho năm 2028, phản ánh thêm một năm phát triển năng lực và một năm kinh nghiệm triển khai. Bạn sẽ thêm, bỏ hoặc nhóm lại những gì?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| CAIS | “Center for AI Safety” | Tổ chức phi lợi nhuận; khung bốn rủi ro; tuyên bố năm 2023 về nguy cơ tuyệt chủng |
| CAISI | “Cơ quan an toàn AI của chính phủ Mỹ” | Trung tâm thuộc NIST; thỏa thuận tự nguyện; đánh giá không mật |
| Four-risk framework | “Cách phân loại của CAIS” | Sử dụng ác ý, cuộc đua AI, rủi ro tổ chức, AI vượt kiểm soát |
| Malicious use | “Tác nhân xấu dùng AI” | Vũ khí sinh học, tin sai lệch, tấn công mạng |
| AI races | “Áp lực cạnh tranh” | Phòng thí nghiệm/công ty/quốc gia thúc đẩy triển khai vượt quá ngưỡng an toàn |
| Organizational risk | “Thất bại nội bộ phòng thí nghiệm” | Văn hóa an toàn, kiểm toán, phòng vệ, an ninh thông tin |
| Rogue AI | “Tác nhân có mục tiêu lệch” | AI có năng lực theo đuổi mục tiêu xung đột với phúc lợi con người |
| California SB-53 | “Quy định cấp bang” | Dự luật 2025–2026; nếu được ký thành luật, sẽ là quy định đầu tiên ở cấp bang của Mỹ về rủi ro thảm họa |

## Đọc thêm

- [Center for AI Safety](https://safe.ai/) — tổ chức xây dựng khung bốn rủi ro.
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — bài viết về bốn rủi ro.
- [CAIS — tuyên bố tháng 5 năm 2023 về nguy cơ tuyệt chủng](https://safe.ai/statement-on-ai-risk) — tuyên bố chung ngắn.
- [NIST CAISI](https://www.nist.gov/caisi) — trung tâm tiêu chuẩn và đổi mới AI thuộc chính phủ.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — liên hệ cam kết ở cấp phòng thí nghiệm với cách nhìn ở quy mô xã hội.

---

[← Bài trước: Chân trời thời gian METR và đánh giá năng lực độc lập](../../21-metr-external-evaluation/docs/vi.md)
