# LLaVA-OneVision: Một ảnh, nhiều ảnh và video trong cùng một mô hình

> Trước LLaVA-OneVision (Li et al., tháng 8/2024), thế giới open-VLM chia thành các dòng riêng: LLaVA-1.5 cho một ảnh (single-image), các mô hình nhiều ảnh (multi-image) như Mantis và VILA, các mô hình video (video) như Video-LLaVA và Video-LLaMA. Mỗi mô hình thắng trên bảng xếp hạng của riêng nó và thất bại ở các bài còn lại. LLaVA-OneVision lập luận rằng một lộ trình huấn luyện (curriculum) duy nhất có thể huấn luyện một mô hình vượt trội ở cả ba kịch bản, và các hiệu ứng chuyển giao tác vụ (task transfer) xuất hiện — kỹ năng một ảnh xuất sang video, suy luận nhiều ảnh xuất sang một ảnh — còn mạnh hơn tổng của các chuyên gia. Công thức đơn giản đến bất ngờ: một ngân sách token thị giác giữ nguyên qua mọi kịch bản, cộng thêm một lộ trình huấn luyện rõ ràng đi từ một ảnh đến OneVision (nhiều ảnh) rồi đến video. Bài học này đọc ngân sách, lộ trình và các hành vi xuất hiện đó.

**Type:** Build
**Languages:** Python (stdlib, token budget solver + curriculum planner)
**Prerequisites:** Giai đoạn 12, Bài 05 (LLaVA), Giai đoạn 12, Bài 06 (độ phân giải tuỳ ý)
**Time:** ~180 phút

## Mục tiêu học tập

- Thiết kế một ngân sách token thị giác giữ nguyên qua đầu vào một ảnh, nhiều ảnh và video.
- Sắp xếp một lộ trình huấn luyện chuyển kỹ năng từ một ảnh sang video mà không quên thảm khốc.
- Giải thích vì sao một mô hình duy nhất đánh bại các chuyên gia cùng số tham số khi lộ trình được làm đúng.
- Kể tên ba năng lực xuất hiện mà LLaVA-OneVision báo cáo: suy luận đa camera, prompt set-of-mark, agent từ ảnh chụp màn hình iPhone.

## Vấn đề

Ảnh, nhiều ảnh và video gây áp lực lên mô hình theo những cách khác nhau.

Một ảnh cần token độ phân giải cao (AnyRes, ~2880 token thị giác) để bắt OCR và chi tiết mịn. Ngân sách mỗi mẫu: một ảnh, 2880 token.

Nhiều ảnh cần vài ảnh ở độ phân giải vừa (~576 token mỗi ảnh) để suy luận xuyên ảnh vừa trong context. Ngân sách mỗi mẫu: 4-8 ảnh, 576 token mỗi ảnh, 2300-4600 token.

Video cần nhiều frame (khung hình) ở độ phân giải thấp, khoảng 196 token mỗi frame sau khi token pooling (gộp token), để bắt động lực thời gian. Ngân sách mỗi mẫu: 8-32 frame, 196 token mỗi frame, 1600-6200 token.

Nếu bạn huấn luyện các mô hình riêng, bạn chọn một ngân sách. Nếu bạn huấn luyện một mô hình, ngân sách phải co giãn hợp lý qua các kịch bản mà không làm tràn context.

Trước OneVision, câu trả lời mặc định là “huấn luyện một kịch bản, bỏ qua các kịch bản còn lại.” Video-LLaVA gắn thêm video vào một mô hình ảnh bằng các giai đoạn huấn luyện phụ. LLaVA-NeXT thêm hỗ trợ nhiều ảnh bằng tiling. Không mô hình nào xử lý cả ba một cách gọn gàng.

## Khái niệm

### Ngân sách token của OneVision

LLaVA-OneVision chọn một ngân sách token thị giác thống nhất khoảng 3000-4000 token mỗi mẫu, phân bổ khác nhau theo từng kịch bản:

- Một ảnh: AnyRes-9 (3x3 ô cộng thumbnail), mỗi ô ở 384 với 729 patch, gộp bilinear mạnh 2x2 → 182 mỗi ô. Tổng: 9 * 182 + 182 = 1820 token. Hoặc AnyRes-4 với 729 mỗi ô = 2916 + 729.
- Nhiều ảnh: mỗi ảnh ở độ phân giải vừa (384, không chia ô), 729 token và không gộp. Ngân sách 6 ảnh → 4374 token.
- Video: 32 frame ở độ phân giải 384 với gộp bilinear 3x3 mạnh → 81 token mỗi frame. Tổng: 32 * 81 = 2592 token.

Cách phân bổ này giữ tổng số token gần như không đổi. LLM không bao giờ nhận một batch làm tràn context. Encoder tạo ra hình học khác nhau theo kịch bản, nhưng LLM dùng cùng một ngân sách.

### Lộ trình huấn luyện ba giai đoạn

LLaVA-OneVision huấn luyện qua ba giai đoạn:

1. SFT một ảnh (giai đoạn SI). Toàn bộ dữ liệu là một ảnh cộng văn bản. Huấn luyện trên đầu vào AnyRes độ phân giải cao. Giai đoạn này dạy nhận thức, OCR và hiểu chi tiết mịn. Dùng dữ liệu LLaVA-NeXT cộng dữ liệu một ảnh riêng của OneVision.
2. SFT OneVision (giai đoạn OV). Trộn một ảnh + nhiều ảnh + video (frame lấy mẫu đều). Huấn luyện trên ngân sách token thống nhất. Giai đoạn này dạy mô hình xử lý các dạng batch không đồng nhất. Không reset trọng số — tiếp tục từ giai đoạn SI.
3. Chuyển giao tác vụ (giai đoạn TT). Tiếp tục với một hỗn hợp tác vụ mục tiêu, thường nặng về nhiều ảnh hoặc video tuỳ sản phẩm. Fine-tune tuỳ chọn cho triển khai.

Quan trọng: thứ tự lộ trình có ý nghĩa. Huấn luyện video trước hoặc nhiều ảnh trước cho hiệu năng ảnh tệ hơn so với một ảnh trước, ngay cả với cùng dữ liệu. Bài báo có ablation rõ ràng cho điều này.

### Vì sao lộ trình lại hiệu quả

Huấn luyện một ảnh xây nền nhận thức. Các patch token mang đặc trưng thị giác chi tiết mịn; LLM học cách tích hợp chúng với văn bản. Nhiều ảnh và video đưa vào những thách thức cấu trúc (ảnh nào là ảnh nào, cái gì xảy ra trước) khó học nếu không có nền nhận thức vững.

Nếu bạn huấn luyện tất cả kịch bản cùng lúc từ đầu, mô hình underfit phần nhận thức (ít dữ liệu một ảnh mỗi batch) và overfit phần cấu trúc (nhiều dữ liệu nhiều ảnh / video). Kết quả: một mô hình bắt được các mẫu suy luận xuyên ảnh nhưng nông về thị giác.

Thứ tự lộ trình cho bạn sức mạnh nhận thức từ giai đoạn SI, rồi suy luận tổ hợp và thời gian từ giai đoạn OV, mà không mất cái nào.

### Kỹ năng xuyên kịch bản xuất hiện

Bài báo LLaVA-OneVision báo cáo ba năng lực xuất hiện:

1. Suy luận đa camera. Được huấn luyện trên nhiều ảnh + video riêng rẽ; khi suy luận, được hỏi về một cảnh lái xe đa camera. Mô hình tích hợp đúng các góc nhìn dù chưa từng thấy định dạng chính xác đó trong huấn luyện.
2. Prompt set-of-mark. Người dùng đánh dấu số lên các vật trong ảnh; mô hình suy luận “dấu 3 đang làm gì so với dấu 7.” Không được huấn luyện trên dấu hay chú thích; học từ sự kết hợp của spatial grounding cộng tham chiếu nhiều ảnh.
3. Agent từ ảnh chụp màn hình iPhone. Người dùng đưa ảnh chụp màn hình iPhone và yêu cầu lập kế hoạch cho lần chạm tiếp theo. Được huấn luyện trên ảnh chụp UI, video quy trình người dùng và các cặp nhiều ảnh trước/sau. Tổng quát hoá sang trường hợp dùng agent.

Đây không phải những tác vụ được huấn luyện; chúng xuất hiện từ cấu trúc tổ hợp của lộ trình.

### Gộp token thị giác

Ngân sách token đòi hỏi phải gộp. OneVision dùng nội suy bilinear trên lưới patch 2D: 24x24 = 576 patch thành 12x12 = 144 (hệ số 2) hoặc 8x8 = 64 (hệ số 3). Việc gộp diễn ra trong không gian lưới patch, không phải không gian token, để giữ tính cục bộ.

Hệ số gộp cho mỗi kịch bản tự nó là một hyperparameter. Gộp ít = nhiều token = biểu diễn phong phú hơn. Gộp nhiều = ít token = chứa được nhiều frame / ảnh hơn.

### LLaVA-OneVision-1.5

Bản tiếp theo năm 2025 (LLaVA-OneVision-1.5, arXiv 2509.23661) “mở hoàn toàn” về dữ liệu huấn luyện, trọng số mô hình và mã nguồn. Nó thu hẹp khoảng cách với các mô hình độc quyền trên một số bảng xếp hạng và phổ biến hoá công thức. Cùng lộ trình, nhiều dữ liệu hơn, LLM nền tốt hơn. Không thay đổi kiến trúc.

### So sánh với Qwen2.5-VL

Qwen2.5-VL (Bài 12.09) đưa ra những lựa chọn khác. Nó dùng M-RoPE và FPS động thay vì gộp cố định. Ngân sách của nó co giãn theo đầu vào — video 1 phút dùng nhiều token hơn video 5 giây. LLaVA-OneVision cố định ngân sách và co giãn việc gộp. Cả hai đều hiệu quả; chúng đánh đổi khả năng cấu hình lấy tính dự đoán được.

```figure
l5-onevision-budget
```

## Áp dụng

`code/main.py` là bộ lập kế hoạch lộ trình và ngân sách cho một VLM kiểu OneVision. Cho một ngân sách token mỗi mẫu và một tỉ lệ kịch bản mục tiêu (ví dụ 40% một ảnh, 30% nhiều ảnh, 30% video), nó:

- Phân bổ độ phân giải, hệ số gộp và số frame cho từng kịch bản.
- Kiểm tra rằng mọi kịch bản đều vừa trong ngân sách chia sẻ.
- Báo cáo số token dự kiến, FLOPs của LLM và kịch bản nào đang thiếu token.
- In ra lịch huấn luyện theo từng giai đoạn.

Dùng nó để lập kế hoạch fine-tune OneVision hoặc để kiểm tra nhanh chi phí mỗi request của một triển khai VLM.

## Đóng gói

Bài học này tạo ra `outputs/skill-onevision-budget-planner.md`. Cho một phân phối tác vụ mục tiêu và một ngân sách mỗi mẫu, nó xuất ra hệ số AnyRes, hệ số gộp mỗi frame, số frame video và trọng số các giai đoạn của lộ trình. Dùng nó mỗi khi bạn huấn luyện hoặc fine-tune một VLM đa kịch bản thống nhất.

## Bài tập

1. Sản phẩm của bạn hỗ trợ 80% một ảnh, 10% nhiều ảnh (2-4 ảnh), 10% video (8-16 frame). Thiết kế ngân sách token. Bạn sẽ đặt phần ngân sách dư — tiết kiệm được nhờ không làm nhiều ảnh nặng — vào đâu?

2. Đọc Mục 4.3 của LLaVA-OneVision (năng lực xuất hiện). Đề xuất một kỹ năng xuất hiện thứ tư mà lộ trình có khả năng mở ra nhưng bài báo không báo cáo.

3. Đảo thứ tự lộ trình — huấn luyện nhiều ảnh trước, rồi một ảnh, rồi video. Dự đoán bảng xếp hạng nào giảm và vì sao.

4. Bài báo báo cáo các bảng xếp hạng video chỉ huấn luyện trên 8 frame mỗi mẫu. Điều đó có tổng quát hoá sang video 30 giây khi suy luận không? Cái gì vỡ trước — ngân sách token hay suy luận thời gian?

5. Gộp bilinear 24x24 patch thành 12x12 là giảm 4 lần mỗi chiều. Cài đặt việc gộp bằng Python stdlib và kiểm chứng rằng trung bình trên mỗi khối 2x2 khớp với kết quả bilinear.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| OneVision scenario | “Một ảnh, nhiều ảnh, hoặc video” | Một trong ba dạng đầu vào mà VLM thống nhất xử lý; ngân sách giữ nguyên qua cả ba |
| Token budget | “Bao nhiêu token mỗi mẫu” | Tổng token thị giác LLM thấy mỗi mẫu huấn luyện / suy luận, thường 3000-4000 |
| Curriculum | “Thứ tự huấn luyện” | Thứ tự các giai đoạn (một ảnh → nhiều ảnh → video) được chọn để có chuyển giao xuất hiện |
| Bilinear pooling | “Thu nhỏ token” | Áp nội suy bilinear lên lưới patch (2D) để giảm số token mà vẫn giữ tính cục bộ |
| Emergent skill | “Không huấn luyện, vẫn làm được” | Năng lực xuất hiện khi suy luận mà không có dữ liệu huấn luyện tương ứng, nhờ cấu trúc lộ trình |
| AnyRes-k | “Thiết lập k ô” | k ô con ở độ phân giải cố định cộng một thumbnail, thường k ∈ {4, 9} |
| Task transfer | “Tổng quát hoá xuyên kịch bản” | Kỹ năng học trên một ảnh áp dụng cho video (và ngược lại) nhờ backbone chia sẻ |

## Đọc thêm

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)

---

[← Bài trước: Công thức cho VLM open-weight (trọng số mở): điều gì thực sự quan trọng](../../07-open-weight-vlm-recipes/docs/vi.md) · [Bài sau: Họ Qwen-VL và Video Dynamic-FPS →](../../09-qwen-vl-family-dynamic-fps/docs/vi.md)
