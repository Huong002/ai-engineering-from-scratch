# Thiết kế tự cải thiện có giới hạn

> Nghiên cứu đã hội tụ vào bốn thành phần nền tảng để giới hạn vòng lặp tự cải thiện. Bất biến hình thức (thuộc tính phải luôn đúng) qua mọi lần chỉnh sửa. Mốc căn chỉnh (mục tiêu cố định) không thể bị sửa. Ràng buộc đa mục tiêu, trong đó mọi khía cạnh (an toàn, công bằng, độ vững) đều phải đạt, không chỉ hiệu năng. Phát hiện suy giảm để tạm dừng vòng lặp khi các chỉ số lịch sử cho thấy năng lực giảm. Không thành phần nào chứng minh được tính an toàn — các kết quả lý thuyết thông tin (độ phức tạp Kolmogorov, định lý Lob) giới hạn điều một hệ thống có thể chứng minh về phiên bản kế nhiệm của chính nó. Chúng là biện pháp giảm thiểu, khiến lỗi âm thầm khó xảy ra hơn.

**Type:** Learn
**Languages:** Python (stdlib, bounded-loop with invariant check)
**Prerequisites:** Giai đoạn 15, Bài 07 (RSI), Giai đoạn 15, Bài 04 (DGM)
**Time:** ~60 phút

## Vấn đề

Mô phỏng cuộc đua ở Bài 7 cho thấy chênh lệch tốc độ nhỏ tích lũy thành khoảng cách lớn. Nghiên cứu tình huống DGM ở Bài 4 cho thấy các vòng lặp có thể chủ động lách bộ đánh giá của mình. Cả hai dẫn đến cùng một câu hỏi kỹ thuật: làm sao ràng buộc vòng lặp tự cải thiện để chính vòng lặp không thể âm thầm nới lỏng các ràng buộc đó?

Bản tóm tắt ICLR 2026 RSI Workshop (openreview.net/pdf?id=OsPQ6zTQXV) nêu bốn thành phần như vậy. RSP v3.0 của Anthropic (Bài 19) và FSF v3 của DeepMind (Bài 20) đều nhắc đến chúng trong các ngưỡng năng lực. Công trình Meta HyperAgents và các khung cộng đồng như SAHOO (tháng 3/2026) triển khai một phần trong thực tế.

Cần nhìn nhận thẳng thắn: đây là các biện pháp giảm thiểu. Các kết quả lý thuyết thông tin giới hạn điều một hệ thống có thể chứng minh về phiên bản kế nhiệm của mình; chưa thiết kế hiện tại nào giải quyết triệt để vấn đề bằng chứng minh hình thức. Vòng lặp được giới hạn tốt an toàn hơn vòng lặp không giới hạn, chứ không an toàn tuyệt đối.

## Khái niệm

### Thành phần 1: bất biến hình thức

Bất biến là thuộc tính phải đúng trước và sau mỗi lần tự sửa đổi. Ví dụ:

- Phân phối đầu ra phụ thuộc vào một phần mở đầu hiến pháp cố định (Bài 17).
- Không lệnh gọi công cụ nào đến điểm cuối chưa được cấp quyền.
- Mọi lần ghi bộ nhớ đều qua đường dẫn có ghi log và chữ ký.
- Hash của mô-đun đánh giá khớp với phiên bản đã duyệt.

Mã bên ngoài mà vòng lặp không thể sửa sẽ kiểm tra các bất biến. Nếu thay đổi đề xuất vi phạm một bất biến, nó bị từ chối. Cách này bắt được kiểu can thiệp bộ đánh giá như DGM (Bài 4).

Khó ở chỗ chọn bất biến vừa cần cho an toàn vừa tính được trong thời gian hợp lý. Đặc tả quá chặt làm đình trệ tiến bộ; quá lỏng để vòng lặp bỏ qua thuộc tính quan trọng.

### Thành phần 2: mốc căn chỉnh

Mốc căn chỉnh là biểu diễn bất biến của mục tiêu cốt lõi, được cố định ngoài phạm vi vòng lặp có thể sửa. Ví dụ:

- Một văn bản hiến pháp (Bài 17) được tải từ vị trí chỉ đọc mỗi lần chạy.
- Một bộ đánh giá ở kho mã riêng, có quyền truy cập độc lập.
- Một danh sách quyền công cụ do con người ký và được kiểm tra lại mỗi chu kỳ.

Mốc này ngăn mục tiêu bị lệch. Vòng lặp có thể cải thiện cách theo đuổi mục tiêu, nhưng không thể sửa chính mục tiêu.

Một kiểu lỗi khó nhận thấy: dù không thể sửa mốc, vòng lặp vẫn có thể diễn giải lại nó khi cách đọc thay đổi. Constitutional AI (Bài 17) chủ ý dựa trên lập luận để xử lý tình huống mới; lớp lập luận đó là nơi cách diễn giải có thể lệch đi. Mốc là cần thiết, nhưng chưa đủ.

### Thành phần 3: ràng buộc đa mục tiêu

Vòng lặp tối ưu một điểm số duy nhất sẽ tìm ra lối tắt. Vòng lặp phải đồng thời đáp ứng nhiều ràng buộc cứng sẽ có ít lối tắt hơn. Các khía cạnh thường gặp:

- Hiệu năng (benchmark ở cấp tác vụ)
- An toàn (đánh giá red-team, tỷ lệ từ chối yêu cầu đã biết là nguy hại)
- Công bằng (giới hạn tác động chênh lệch lên các nhóm nhạy cảm)
- Độ vững (bộ kiểm thử ngoài phân phối, xử lý đầu vào đối kháng)

Một thay đổi chỉ được chấp nhận nếu mọi ràng buộc đều đạt. Bộ điều tiết chi phí ở Bài 13 bổ sung ràng buộc tài chính. Llama Guard ở Bài 18 đóng vai trò khía cạnh an toàn.

### Thành phần 4: phát hiện suy giảm

Điểm của mỗi chu kỳ được so với phân phối lịch sử. Nếu mức giảm vượt ngưỡng cho phép đã đặt, vòng lặp tạm dừng. Cách này bắt được sự suy giảm năng lực âm thầm, vốn có thể bị trung bình động hấp thụ khi vòng lặp “cải thiện vượt qua” nó.

Một cách triển khai thực tế: lưu điểm từng tác vụ của N chu kỳ gần nhất. Mỗi chu kỳ mới tính mức chênh lệch cho từng tác vụ. Nếu bất kỳ mức chênh lệch nào thấp hơn ngưỡng, chu kỳ bị từ chối và con người xem xét.

### Giới hạn lý thuyết thông tin

Độ phức tạp Kolmogorov và định lý Lob đặt giới hạn trên cho điều một hệ thống có thể chứng minh về chính mình. Godel Machine hình thức của Schmidhuber (Bài 4) hướng tới giới hạn cao nhất đó; chưa ai hoàn thành một chứng minh không tầm thường. Kết quả của Lob nói rằng: nếu một hệ thống chứng minh được rằng nó tin “Tôi sẽ làm X nếu chứng minh được mình nên làm X”, nó sẽ làm X mà không chứng minh mình nên làm; đây là lỗi tự tham chiếu đã biết.

Hệ quả với các thành phần của ta: chúng không thể giải quyết triệt để vấn đề an toàn. Chúng khiến lỗi âm thầm khó xảy ra hơn. Một vòng lặp có ác ý hoặc đã lệch hướng, thay vì lặng lẽ vượt qua kiểm tra còn thiếu, giờ phải chủ động phá một kiểm tra hiện hữu — dấu hiệu dễ phát hiện hơn.

### Ví dụ minh họa

Giả sử một agent đề xuất chỉnh sửa. Các lớp kiểm soát gồm:

1. Kiểm tra bất biến: hash mô-đun, danh sách quyền công cụ, phần mở đầu hiến pháp.
2. Kiểm tra mốc: tuyên bố mục tiêu khớp phiên bản đã duyệt (theo byte hoặc ngữ nghĩa).
3. Đánh giá đa mục tiêu: các khía cạnh hiệu năng, an toàn, công bằng, độ vững.
4. Phát hiện suy giảm: không khía cạnh nào giảm quá ngưỡng cho phép.

Cả bốn phải đạt thì thay đổi mới được áp dụng. Chỉ một kiểm tra không đạt cũng làm vòng lặp tạm dừng.

```figure
bounded-gates
```

## Áp dụng

`code/main.py` chạy vòng lặp tự cải thiện có giới hạn trên ví dụ mô phỏng kiểu DGM ở Bài 4, nhưng thêm bốn thành phần trên. Bạn có thể bật hoặc tắt từng thành phần riêng. Phần minh họa cho thấy mỗi thành phần bắt một loại lỗi cụ thể, và bỏ bất kỳ thành phần nào cũng để loại lỗi đó lọt qua.

## Đóng gói

`outputs/skill-bounded-loop-review.md` kiểm tra một vòng lặp có giới hạn được đề xuất và chấm xem nó thực sự triển khai thành phần nào trong bốn thành phần, thay vì chỉ tuyên bố có.

## Bài tập

1. Chạy `code/main.py` khi bật mọi thành phần. Xác nhận vòng lặp vẫn cải thiện chỉ số chính mà không để cách lách luật thắng.

2. Tắt phát hiện suy giảm. Tạo đầu vào khiến suy giảm năng lực âm thầm được chấp nhận.

3. Tắt ràng buộc đa mục tiêu. Cho thấy vòng lặp hội tụ theo khía cạnh hiệu năng trong khi khía cạnh an toàn giảm.

4. Thiết kế mốc căn chỉnh cho agent viết mã. Dùng văn bản gì, lưu ở đâu, kiểm tra thế nào?

5. Đọc bản tóm tắt ICLR 2026 RSI Workshop. Chọn một trong bốn thành phần và đề xuất cải tiến cụ thể so với trình độ hiện tại.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Invariant | “Thuộc tính luôn đúng” | Thuộc tính được mã bên ngoài kiểm tra trước và sau mỗi lần chỉnh sửa |
| Alignment anchor | “Mục tiêu cố định” | Biểu diễn mục tiêu cốt lõi bất biến, nằm ngoài phạm vi vòng lặp có thể sửa |
| Multi-objective constraint | “Mọi khía cạnh đều phải đạt” | Hiệu năng, an toàn, công bằng, độ vững — đều bắt buộc |
| Regression detection | “Giảm thì dừng” | Tạm dừng vòng lặp khi mức chênh lệch chỉ số lịch sử cho thấy năng lực giảm |
| Kolmogorov bound | “Giới hạn lý thuyết thông tin” | Giới hạn điều hệ thống có thể chứng minh về phiên bản kế nhiệm của mình |
| Lob's theorem | “Bẫy tự tham chiếu” | Hệ thống có thể hành động theo “mình nên làm” mà không chứng minh điều đó |
| Gate stack | “Kiểm tra nhiều lớp” | Kết hợp nhiều thành phần; một lỗi cũng đủ từ chối thay đổi |
| Bounded improvement | “Giảm thiểu, không phải chứng minh” | Khiến lỗi âm thầm khó xảy ra hơn; không giải quyết triệt để vấn đề an toàn |

## Đọc thêm

- [Tóm tắt ICLR 2026 RSI Workshop (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) — sự hội tụ vào bốn thành phần.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — các ngưỡng năng lực đa mục tiêu.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — giám sát căn chỉnh giả dối như một thành phần bất biến.
- [Schmidhuber (2003). Godel Machines](https://people.idsia.ch/~juergen/goedelmachine.html) — tiền thân dựa trên chứng minh hình thức của các thành phần này.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — mốc căn chỉnh dựa trên lập luận.

---

[← Bài trước: Tự cải thiện đệ quy](../../07-recursive-self-improvement/docs/vi.md) · [Bài sau: Toàn cảnh agent lập trình tự chủ (2026) →](../../09-coding-agent-landscape/docs/vi.md)
