# Lập kế hoạch với HTN (hierarchical task network - mạng tác vụ phân cấp) và Tìm kiếm Tiến hoá

> Lập kế hoạch tượng trưng xử lý những trường hợp mà kế hoạch chứng minh được là đúng. Tìm kiếm mã tiến hoá xử lý những trường hợp mà hàm độ thích nghi kiểm tra được bằng máy. ChatHTN (2025) và AlphaEvolve (2025) cho thấy mỗi bên mở ra điều gì khi ghép với một LLM.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 02 (ReWOO và plan-and-execute)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích Hierarchical Task Networks: tác vụ, phương thức, operator (toán tử), điều kiện tiên quyết, hiệu ứng.
- Mô tả vòng lặp lai của ChatHTN — tìm kiếm tượng trưng với decomposition (phân rã) dự phòng do LLM đề xuất.
- Giải thích vòng lặp evolutionary (tiến hoá) của AlphaEvolve và lý do nó chỉ hoạt động khi có bộ đánh giá lập trình được.
- Cài đặt một HTN planner đồ chơi cùng một tìm kiếm tiến hoá đồ chơi bằng stdlib.

## Vấn đề

ReWOO (Bài 02), Plan-and-Execute và ReAct bao phủ phần lớn việc lập kế hoạch của agent. Có hai trường hợp chúng chưa xử lý tốt:

1. **Kế hoạch có tính đúng đắn chứng minh được.** Lập lịch, định tuyến bay, quy trình tuân thủ — kế hoạch phải đúng ngay từ cách xây dựng. Một kế hoạch LLM trôi chảy nhưng thỉnh thoảng ảo giác ra một bước là không chấp nhận được.
2. **Tối ưu hoá với hàm độ thích nghi kiểm tra được bằng máy.** Nhân ma trận, heuristic lập lịch, lượt biên dịch — mục tiêu không phải "một kế hoạch đúng" mà là "kế hoạch tốt nhất".

Lập kế hoạch HTN và AlphaEvolve giải hai bài toán khác nhau. Cả hai đều dùng LLM như bộ khuếch đại, không phải vật thay thế.

## Khái niệm

### Mạng tác vụ phân cấp

Một HTN gồm:

- **Tác vụ** — phức hợp (cần phân rã) và nguyên thuỷ (thực thi trực tiếp).
- **Phương thức** — cách phân rã một tác vụ phức hợp thành các tác vụ con, kèm điều kiện tiên quyết.
- **Toán tử** — hành động nguyên thuỷ với điều kiện tiên quyết và hiệu ứng.
- **Trạng thái** — một tập sự kiện.

Lập kế hoạch: cho một tác vụ mục tiêu và một trạng thái ban đầu, tìm một cách phân rã thành các toán tử nguyên thuỷ mà điều kiện tiên quyết của chúng được thoả mãn lần lượt.

HTN ra đời trước LLM và vẫn là chuẩn tham chiếu cho các kế hoạch đúng đắn chứng minh được.

### ChatHTN (Gopalakrishnan và cộng sự, 2025)

ChatHTN (arXiv:2505.11814) xen kẽ HTN tượng trưng với các truy vấn LLM:

1. Thử phân rã tác vụ phức hợp hiện tại bằng các phương thức đã có.
2. Nếu không phương thức nào áp dụng được, hỏi LLM: "bạn sẽ phân rã `task` trong trạng thái `s` như thế nào?"
3. Chuyển câu trả lời của LLM thành các tác vụ con ứng viên.
4. Kiểm tra với lược đồ toán tử; loại bỏ những cách phân rã không hợp lệ.
5. Đệ quy.

Tuyên bố trung tâm của bài báo: mọi kế hoạch sinh ra đều đúng đắn chứng minh được vì gợi ý của LLM chỉ vào như một cách phân rã ứng viên, không bao giờ là chỉnh sửa kế hoạch trực tiếp. Lớp tượng trưng giữ trách nhiệm đúng đắn; LLM mở rộng thư viện phương thức.

Online method learning (OpenReview `gwYEDY9j2x`, bài tiếp theo 2025) thêm một bộ học tổng quát hoá các cách phân rã do LLM tạo ra bằng hồi quy — cắt tần suất truy vấn LLM tới 75%.

### AlphaEvolve (Novikov và cộng sự, 2025)

AlphaEvolve (arXiv:2506.13131, DeepMind, tháng 6 năm 2025) là một con thú khác: tìm kiếm mã tiến hoá do một tập hợp Gemini 2.0 Flash/Pro điều phối.

Vòng lặp:

1. Bắt đầu với một chương trình hạt giống + một bộ đánh giá lập trình được (trả về điểm thích nghi).
2. Tập hợp các LLM đề xuất đột biến.
3. Chạy các đột biến qua bộ đánh giá.
4. Giữ bản tốt nhất; đột biến tiếp.

Những thắng lợi đã công bố:

- Lần cải thiện đầu tiên so với Strassen cho phép nhân ma trận phức 4x4 sau 56 năm (48 phép nhân vô hướng).
- Thu hồi 0,7% compute của Google nhờ một heuristic lập lịch Borg.
- Tăng tốc FlashAttention 32% trên một tải tiên phong.

Ràng buộc cứng: hàm độ thích nghi phải kiểm tra được bằng máy. Tìm kiếm tiến hoá trên câu trả lời bằng văn xuôi sẽ không hội tụ.

### Khi nào dùng cái nào

| Lớp bài toán | Dùng | Vì sao |
|---------------|-----|-----|
| Lập lịch với ràng buộc cứng | HTN + ChatHTN | Đúng đắn chứng minh được |
| Tối ưu hoá biên dịch | AlphaEvolve | Hàm thích nghi kiểm tra được bằng máy |
| Thực thi tác vụ nhiều bước | ReAct / ReWOO | LLM trong vòng lặp, không có bảo đảm hình thức |
| Cải thiện mã có kiểm thử | AlphaEvolve | Kiểm thử là bộ đánh giá |
| Tự động hoá ràng buộc chính sách | HTN | Điều kiện tiên quyết mã hoá chính sách |

### Chỗ mẫu hình này đi sai

- **HTN không có toán tử.** Không có lược đồ điều kiện tiên quyết/hiệu ứng thì tuyên bố đúng đắn sụp đổ. "LLM đề xuất cách phân rã" của ChatHTN cần lược đồ để loại bỏ nước đi không hợp lệ.
- **AlphaEvolve không có bộ đánh giá thật.** "Hỏi LLM xem mã có tốt hơn không" không phải là hàm thích nghi. Bộ đánh giá phải tất định và nhanh.
- **Làm quá mức cần thiết.** Hầu hết tác vụ agent không cần cả hai. Hãy dùng ReAct hoặc ReWOO trước.

```figure
htn-tree-expand
```

## Tự tay làm

`code/main.py` triển khai hai món đồ chơi:

- Một HTN planner stdlib với toán tử, phương thức, điều kiện tiên quyết, hiệu ứng, và một `LLMFallback` kích hoạt khi không phương thức nào khớp với một tác vụ phức hợp. "LLM" ở đây là bộ phân rã theo kịch bản để planner chạy ngoại tuyến.
- Một tìm kiếm tiến hoá stdlib trên các chương trình số học: mọc các biểu thức có đầu ra cực tiểu hoá `|f(x) - target|` trên một tập kiểm thử. Bộ đánh giá là tất định.

Chạy nó:

```
python3 code/main.py
```

Dấu vết cho thấy HTN planner phân rã một tác vụ phức hợp (có dự phòng LLM giữa kế hoạch) và vòng lặp tiến hoá hội tụ về một biểu thức mục tiêu.

## Áp dụng

- **HTN planner** — `pyhop`, `SHOP3`, hoặc tự xây cho việc thực thi chính sách theo miền cụ thể.
- **ChatHTN** — mã nghiên cứu; mẫu hình (tượng trưng + dự phòng LLM) chuyển được sạch sang bất kỳ HTN planner nào.
- **AlphaEvolve** — bài báo DeepMind; mẫu hình (tập hợp + bộ đánh giá) tái tạo được. OpenEvolve và các bản fork mã nguồn mở tương tự đang nổi lên.
- **Framework agent** — chưa có cái nào cung cấp HTN hay AlphaEvolve sẵn. Hãy xây nó như một subagent hoặc một worker chạy nền.

## Đóng gói

`outputs/skill-hybrid-planner.md` sinh ra một bộ khung planner lai (HTN hoặc tiến hoá) với vai trò LLM được giới hạn tường minh.

## Bài tập

1. Mở rộng HTN planner với quay lui: khi hậu điều kiện của một toán tử thất bại lúc chạy, quay lại và thử phương thức tiếp theo.
2. Thêm cache phương thức LLM cho ChatHTN: khi LLM phân rã tác vụ `T` trong mẫu trạng thái `P`, lưu kết quả. Lần gọi sau kiểm tra thư viện phương thức trước.
3. Đổi bộ đánh giá của tìm kiếm tiến hoá sang một bộ kiểm thử thật. Tiến hoá một hàm sắp xếp vượt 20 ca kiểm thử; báo cáo số thế hệ tới khi hội tụ.
4. Đọc ghi chú thiết kế bộ đánh giá của AlphaEvolve. Thiết kế một bộ đánh giá cho miền bạn quan tâm (tối ưu truy vấn SQL, thu gọn bộ kiểm thử, YAML triển khai).
5. Kết hợp: dùng HTN phân rã một tác vụ phức hợp thành các tác vụ con, rồi dùng tìm kiếm tiến hoá trên toán tử nguyên thuỷ của từng tác vụ con. Chỗ nào nó toả sáng, chỗ nào nó làm quá mức?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| HTN | "Planner phân cấp" | Phân rã tác vụ với toán tử, điều kiện tiên quyết, hiệu ứng |
| Method | "Quy tắc phân rã" | Cách chia một tác vụ phức hợp thành các tác vụ con |
| Operator | "Hành động nguyên thuỷ" | Bước cụ thể với điều kiện tiên quyết và hiệu ứng |
| ChatHTN | "LLM + HTN" | Planner tượng trưng hỏi LLM khi không phương thức nào khớp |
| AlphaEvolve | "Tìm kiếm mã tiến hoá" | Tập hợp LLM đột biến mã; bộ đánh giá tất định chọn |
| Fitness function | "Bộ đánh giá" | Điểm tất định, kiểm tra được bằng máy, trên các đầu ra |
| Online method learning | "Cache cách phân rã của LLM" | Lưu + tổng quát hoá kế hoạch LLM để cắt chi phí truy vấn |

## Đọc thêm

- [Gopalakrishnan và cộng sự, ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — planner lai tượng trưng + LLM
- [Novikov và cộng sự, AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — tìm kiếm mã tiến hoá với đột biến do LLM
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — khi nào nên dùng planner thay vì một vòng lặp đơn giản
