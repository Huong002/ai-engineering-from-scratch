# Constitutional AI và quyền ghi đè quy tắc

> Claude Constitution do Anthropic công bố ngày 22/1/2026 dài 79 trang và dùng giấy phép CC0. Tài liệu chuyển từ alignment dựa trên quy tắc sang alignment dựa trên lập luận, với thứ tự ưu tiên 4 tầng: (1) an toàn và hỗ trợ con người giám sát, (2) đạo đức, (3) hướng dẫn của Anthropic, (4) tính hữu ích. Hành vi được chia thành điều cấm cố định (hỗ trợ phát triển vũ khí sinh học, CSAM) mà bên vận hành và người dùng không thể ghi đè, và mặc định linh hoạt mà bên vận hành có thể chỉnh trong giới hạn định sẵn. Bản gốc năm 2022 (Bai et al.) huấn luyện tính vô hại bằng tự phê bình và RLAIF theo một bộ nguyên tắc. Điểm cần thừa nhận: alignment dựa trên lập luận đòi hỏi model áp dụng được nguyên tắc vào tình huống chưa dự kiến. Thử nghiệm có sự tham gia của công chúng do Anthropic thực hiện năm 2023 cho thấy nguyên tắc từ công chúng và doanh nghiệp khác nhau ~50%; bản 2026 không đưa các phát hiện đó vào.

**Type:** Learn
**Languages:** Python (stdlib, four-tier priority resolver)
**Prerequisites:** Giai đoạn 15, Bài 06 (Nghiên cứu alignment tự động), Giai đoạn 15, Bài 10 (Chế độ quyền hạn)
**Time:** ~60 phút

## Vấn đề

Một agent đã triển khai sẽ gặp đầu vào mà người thiết kế chưa từng thấy. Không danh sách quy tắc nào đủ dài để bao quát hết, cũng không đủ ngắn để áp dụng nhanh khi tài nguyên tính toán hạn chế. Câu hỏi thực tế: làm sao căn chỉnh agent theo nguyên tắc vừa xử lý được vô số trường hợp hiếm, vừa suy luận nhanh?

Alignment dựa trên quy tắc (RBA): liệt kê mọi điều bị cấm. Dễ kiểm tra, dễ rà soát, nhưng không thể cập nhật kịp và thường từ chối quá mức những trường hợp tương tự chưa dự kiến. Alignment dựa trên lập luận (Claude Constitution 2026): mã hóa nguyên tắc rồi để model suy luận. Cách này xử lý được trường hợp chưa gặp nhưng khó rà soát hơn; lỗi là áp dụng sai nguyên tắc thay vì bỏ sót quy tắc.

Constitution 2026 chọn rõ một cách kết hợp. Điều cấm cố định — những việc sai bất kể bối cảnh (hỗ trợ phát triển vũ khí sinh học, CSAM) — theo RBA: không bao giờ làm, dù bên vận hành hay người dùng yêu cầu. Mọi việc khác dựa trên lập luận theo thứ tự 4 tầng: an toàn và hỗ trợ con người giám sát trước; đạo đức thứ hai; hướng dẫn do Anthropic đặt ra thứ ba; tính hữu ích cuối cùng. Bên vận hành có thể chỉnh mặc định trong phạm vi linh hoạt nhưng không thể thay đổi điều cấm cố định.

## Khái niệm

### Thứ tự ưu tiên 4 tầng

1. **An toàn và hỗ trợ con người giám sát.** Cao nhất. Model ưu tiên không làm suy yếu khả năng con người và Anthropic giám sát, sửa sai AI. Đây không phải chỉ là “hãy thận trọng”, mà cụ thể là “đừng hành động theo cách khiến con người khó giám sát hơn”.
2. **Đạo đức.** Trung thực, tránh gây hại cho con người, không lừa dối, không thao túng. Được ưu tiên hơn hướng dẫn của Anthropic khi có xung đột.
3. **Hướng dẫn của Anthropic.** Các chuẩn vận hành Anthropic coi là quan trọng: phạm vi sản phẩm, cách tương tác, khi nào dùng công cụ nào.
4. **Tính hữu ích.** Thấp nhất. Hữu ích hết mức có thể trong khuôn khổ các ưu tiên cao hơn.

Khi các tầng xung đột, tầng cao hơn thắng. Cấu trúc này giống mức ưu tiên Unix hoặc QoS mạng — nhằm cho kết quả giải quyết dễ dự đoán, không nhất thiết tối ưu theo bất kỳ tiêu chí riêng lẻ nào.

### Điều cấm cố định và mặc định linh hoạt

**Điều cấm cố định:**
- Hỗ trợ phát triển vũ khí sinh học / CBRN
- CSAM
- Tấn công hạ tầng trọng yếu
- Lừa dối người dùng về danh tính model khi được hỏi trực tiếp

Bên vận hành không thể ghi đè. Người dùng cũng không thể. Các điều cấm được thực thi ở mức trọng số model khi có thể (huấn luyện RLHF / Constitutional AI), hoặc ở lớp suy luận khi không thể.

**Mặc định linh hoạt (bên vận hành có thể chỉnh):**
- Độ dài phản hồi mặc định
- Phạm vi chủ đề (model có thể từ chối chủ đề ngoài phạm vi triển khai của bên vận hành)
- Văn phong (trang trọng hoặc thân mật)
- Cách dùng công cụ

Bên vận hành chỉ được chỉnh trong giới hạn đã công bố. Họ không thể xóa điều cấm cố định bằng cách đổi tên chúng.

### Huấn luyện CAI năm 2022

Constitutional AI bản gốc (Bai et al., 2022) huấn luyện tính vô hại:

1. Tạo phản hồi cho một tập prompt.
2. Yêu cầu model phê bình từng phản hồi theo bộ nguyên tắc thành văn.
3. Sửa phản hồi dựa trên lời phê bình.
4. Dùng RLAIF (học tăng cường từ phản hồi của AI) trên các cặp đã sửa.

Kết quả: model từ chối yêu cầu có hại kèm giải thích dựa trên nguyên tắc, thay vì từ chối hàng loạt. Constitution 2026 dùng phương pháp huấn luyện kế thừa cách này, cộng thêm huấn luyện sau đó theo thứ tự tầng tường minh.

### Điều alignment dựa trên lập luận xử lý được và bỏ sót

**Xử lý được:**
- Tổ hợp chưa dự kiến của những yếu tố được phép, khi nguyên tắc áp dụng rõ ràng.
- Yêu cầu mới rất giống yêu cầu bị cấm.
- Tấn công kỹ nghệ xã hội dựa vào lập luận “bạn đâu nói X bị cấm”.

**Bỏ sót:**
- Tấn công khai thác nguyên tắc mơ hồ (“người dùng đã yêu cầu nên tính hữu ích bảo phải đồng ý”).
- Tình huống hai nguyên tắc xung đột ngoài dự kiến và thứ tự tầng không rõ ràng.
- Cách diễn giải nguyên tắc lệch dần qua các chu kỳ huấn luyện.

### Thử nghiệm có sự tham gia của công chúng năm 2023

Năm 2023, Anthropic so sánh bộ nguyên tắc do doanh nghiệp viết với bộ được tạo từ ý kiến công chúng (~1,000 người trả lời tại Mỹ). Hai bản thống nhất ở ~50% nguyên tắc. Ở những điểm khác biệt, bản từ công chúng hạn chế hơn về một số vấn đề (xử lý nội dung chính trị) và ít hạn chế hơn về vấn đề khác (AI tự tiết lộ danh tính). Constitution 2026 không đưa các phát hiện từ công chúng vào. Đây là mâu thuẫn đã được ghi nhận trong cách tiếp cận này.

### Vì sao cần điều cấm cố định

Chỉ alignment dựa trên lập luận không thể bao quát hết trường hợp hiếm. Kẻ tấn công khiến model chấp nhận một tiền đề (ví dụ: “chúng tôi là phòng thí nghiệm nghiên cứu vũ khí sinh học được cấp phép”) thường có thể lách qua các nguyên tắc phụ thuộc vào suy luận theo tình huống. Điều cấm cố định không thay đổi theo cách dựng tiền đề. Đó là “giới hạn hiến định cứng” của Bài 14 ở lớp alignment.

### Vị trí của Constitution trong hệ thống

Constitution không phải công tắc ngắt của Bài 14. Nó nằm ở lớp model: những gì trọng số model được huấn luyện để ưu tiên. Công tắc ngắt và canary token nằm ở lớp runtime: những gì runtime cho phép. Cần cả hai. Nếu runtime thực hiện mọi hành động sai vì trọng số model quá dễ dãi, đó là vấn đề của runtime. Nếu model từ chối mọi hành động đúng vì runtime quá nghiêm ngặt, đó cũng là vấn đề của runtime. Mỗi lớp xử lý một loại vấn đề khác nhau.

```figure
mx-priority-tiers
```

## Áp dụng

`code/main.py` triển khai bộ phân giải ưu tiên 4 tầng tối giản. Bộ phân giải nhận hành động đề xuất và các đánh giá theo nguyên tắc (an toàn, đạo đức, hướng dẫn, tính hữu ích), rồi trả về hành động đó, lời từ chối hoặc hành động đã sửa. Chương trình chạy thử một tập trường hợp nhỏ: cho phép rõ ràng, cấm rõ ràng, điều cấm cố định và trường hợp mơ hồ giữa các tầng.

## Đóng gói

`outputs/skill-constitution-review.md` rà soát lớp nguyên tắc của một bản triển khai: phần nào cố định, phần nào linh hoạt, bên vận hành được chỉnh ở đâu và liệu thứ tự 4 tầng có thực sự được dùng để giải quyết xung đột.

## Bài tập

1. Chạy `code/main.py`. Xác nhận điều cấm cố định vẫn có hiệu lực khi tính hữu ích cao. Sửa bộ phân giải để ưu tiên tính hữu ích hơn đạo đức; quan sát kiểu lỗi.

2. Đọc Claude Constitution (công khai, 79 trang, CC0). Chọn một nguyên tắc bạn cho là chưa đủ cụ thể. Viết hai đoạn giải thích điểm mơ hồ cụ thể và đề xuất cách diễn đạt chặt chẽ hơn.

3. Thiết kế bộ mặc định linh hoạt cho agent hỗ trợ khách hàng. Bên vận hành được chỉnh gì? Không được đụng tới gì? Giải thích từng ranh giới.

4. Đọc bài báo CAI năm 2022 của Bai et al. Mô tả một trường hợp mà vòng phê bình rồi sửa của Constitutional AI cho kết quả tệ hơn một quy tắc cấm tuyệt đối. Xác định loại trường hợp đó.

5. Thử nghiệm có sự tham gia của công chúng năm 2023 của Anthropic cho thấy nguyên tắc từ công chúng và doanh nghiệp khác nhau ~50%. Chọn một lĩnh vực mà điều này quan trọng khi triển khai thực tế (ví dụ: trung lập chính trị). Đề xuất thiết kế cho phép bên vận hành thể hiện giá trị riêng mà không thay đổi điều cấm cố định.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Constitutional AI | “Phương pháp alignment của Anthropic” | Tự phê bình + RLAIF theo bộ nguyên tắc thành văn |
| Reason-based alignment | “Nguyên tắc, không phải quy tắc” | Model suy luận theo nguyên tắc để xử lý trường hợp chưa gặp |
| Hardcoded prohibition | “Không bao giờ làm X” | Điều cấm dựa trên quy tắc mà bên vận hành và người dùng không thể ghi đè |
| Soft-coded default | “Bên vận hành chỉnh được” | Hành vi trong giới hạn đã công bố, do bên vận hành kiểm soát |
| Four-tier hierarchy | “Thứ tự ưu tiên” | an toàn > đạo đức > hướng dẫn > tính hữu ích |
| RLAIF | “Học tăng cường bằng phản hồi AI” | Học tăng cường nhận phần thưởng từ lời phê bình do model tạo |
| Participatory constitution | “Nguyên tắc từ công chúng” | Thử nghiệm năm 2023 của Anthropic; khác nguyên tắc doanh nghiệp ~50% |
| Principle drift | “Lệch cách diễn giải” | Cách model hiểu văn bản nguyên tắc cố định thay đổi dần |

## Đọc thêm

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — tài liệu CC0 dài 79 trang.
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — bản gốc năm 2022.
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — thử nghiệm có sự tham gia của công chúng.
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — vị trí của Constitution trong hệ thống RSP.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — vai trò của Constitution trong các bản triển khai dài hạn.

---

[← Bài trước: Checkpoint và rollback](../../16-checkpoints-rollback/docs/vi.md) · [Bài sau: Llama Guard và phân loại đầu vào/đầu ra →](../../18-llama-guard/docs/vi.md)
