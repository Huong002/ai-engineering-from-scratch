# Benchmarks: WebArena và OSWorld

> WebArena kiểm tra năng lực agent web trên bốn ứng dụng tự host. OSWorld kiểm tra năng lực agent desktop trên Ubuntu, Windows, macOS. Khi công bố (2023–2024), cả hai cho thấy khoảng cách lớn giữa agent tốt nhất và con người. Khoảng cách đang thu hẹp; các kiểu thất bại thì không đổi.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 19 (SWE-bench, GAIA)
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả bốn ứng dụng tự host của WebArena và vì sao đánh giá dựa trên thực thi lại quan trọng.
- Giải thích vì sao OSWorld dùng ảnh chụp màn hình hệ điều hành thật thay vì API trợ năng.
- Kể tên hai kiểu thất bại chính của OSWorld: GUI grounding và kiến thức vận hành.
- Tóm tắt OSWorld-G và OSWorld-Human bổ sung gì so với benchmark gốc.

## Vấn đề

Agent đa dụng có thể gọi công cụ. Liệu chúng có thể lái trình duyệt qua 20 lần click để hoàn tất thanh toán mua hàng? Liệu chúng có thể cấu hình một máy Linux chỉ bằng bàn phím và chuột? Đó là những câu hỏi WebArena và OSWorld trả lời.

## Khái niệm

### WebArena (Zhou et al., ICLR 2024)

- 812 nhiệm vụ dài hạn trên bốn ứng dụng web tự host: một trang mua sắm, một diễn đàn, một công cụ dev giống GitLab, một CMS doanh nghiệp.
- Cộng thêm tiện ích: bản đồ, máy tính, scratchpad.
- Đánh giá dựa trên thực thi qua gym API — đơn hàng đã được đặt chưa, issue đã đóng chưa, trang CMS đã cập nhật chưa?
- Khi công bố: agent GPT-4 tốt nhất đạt 14.41% tỉ lệ thành công so với con người 78.24%.

Cách tự host quan trọng — benchmark không chập chờn vì các ứng dụng đích được ghim phiên bản và tái lập được.

### Phần mở rộng

- **VisualWebArena** — nhiệm vụ neo vào thị giác, nơi thành công phụ thuộc vào việc diễn giải hình ảnh (ảnh chụp màn hình là quan sát hạng nhất).
- **TheAgentCompany** (12/2024) — thêm terminal + coding; giống một environment (môi trường) làm việc từ xa thật hơn.

### OSWorld (Xie et al., NeurIPS 2024)

- 369 nhiệm vụ máy tính thật trên Ubuntu, Windows, macOS.
- Điều khiển bàn phím và chuột tự do trên ứng dụng thật.
- Ảnh chụp màn hình 1920×1080 làm quan sát.
- Khi công bố: model tốt nhất 12.24% so với con người 72.36%.

### Hai kiểu thất bại chính

1. **GUI grounding.** Ánh xạ pixel → phần tử. Model gặp khó khi định vị phần tử UI một cách đáng tin cậy ở 1920×1080.
2. **Kiến thức vận hành.** Menu nào có cài đặt đó, phím tắt nào, ngăn tùy chọn nào. Phần đuôi kiến thức mà con người tích lũy qua nhiều năm.

### Các nghiên cứu tiếp nối

- **OSWorld-G** — bộ grounding 564 mẫu + bộ huấn luyện Jedi. Tách grounding khỏi lập kế hoạch để bạn đo riêng từng thứ.
- **OSWorld-Human** — quỹ đạo hành động vàng được tuyển chọn thủ công. Cho thấy các agent hàng đầu dùng số bước nhiều hơn 1.4-2.7x so với mức cần thiết (khoảng cách hiệu quả quỹ đạo).

### Vì sao điều này quan trọng

Claude computer use, OpenAI CUA, Gemini 2.5 Computer Use (Bài 21) đều huấn luyện trên khối lượng công việc được định hình bởi WebArena và OSWorld. Benchmark là mục tiêu; model sản xuất là câu trả lời đã giao.

### Chỗ đánh giá benchmark hay sai

- **Đánh giá chỉ bằng ảnh chụp màn hình.** OSWorld được điều khiển bằng ảnh chụp màn hình; đánh giá một agent dùng DOM hoặc API trợ năng trên OSWorld sẽ bỏ sót thử thách grounding.
- **Bỏ qua độ dài quỹ đạo.** Chỉ chấm success rate (tỉ lệ thành công) sẽ bỏ sót mức kém hiệu quả 1.4-2.7x số bước mà OSWorld-Human phát hiện.
- **Ứng dụng tự host cũ.** Ứng dụng của WebArena ghim phiên bản cụ thể; cập nhật mà không tuyển chọn lại sẽ phá vỡ khả năng so sánh.

```figure
ae-agent-human-gap
```

## Tự tay làm

`code/main.py` triển khai một harness agent web đồ chơi:

- Một máy trạng thái "shopping app" tối giản: list_items, add_to_cart, checkout.
- Quỹ đạo vàng cho 3 nhiệm vụ.
- Một agent kịch bản thử từng nhiệm vụ.
- Bộ đánh giá dựa trên thực thi (kiểm tra trạng thái) và metric hiệu quả quỹ đạo (số bước so với vàng).

Chạy thử:

```
python3 code/main.py
```

Output: tỉ lệ thành công theo từng nhiệm vụ và hiệu quả quỹ đạo, theo đúng phương pháp luận của OSWorld-Human.

## Áp dụng

- **WebArena Verified** tự host trên một cluster nội bộ để đánh giá liên tục.
- **OSWorld** trong một cụm VM cho agent desktop.
- **Agent dùng máy tính** (Bài 21) — Claude, OpenAI CUA, Gemini — đều huấn luyện trên khối lượng công việc như thế.
- **Luồng sản phẩm của chính bạn** — ghi lại quỹ đạo vàng cho 20 nhiệm vụ hàng đầu; chạy agent đối chiếu hàng tuần.

## Đóng gói

`outputs/skill-web-desktop-harness.md` xây một harness agent web/desktop với đánh giá dựa trên thực thi và metric hiệu quả quỹ đạo.

## Bài tập

1. Mở rộng harness đồ chơi với ứng dụng thứ hai (một diễn đàn). Viết 3 nhiệm vụ kèm quỹ đạo vàng.
2. Thêm báo cáo hiệu quả quỹ đạo theo từng nhiệm vụ. Trên đồ chơi của bạn, agent là 1x, 2x hay 3x so với vàng?
3. Triển khai một công cụ "distractor" — công cụ mà quỹ đạo vàng không bao giờ dùng. Agent kịch bản có bị cám dỗ không?
4. Đọc OSWorld-G. Bạn sẽ tách thất bại grounding khỏi thất bại lập kế hoạch trong đánh giá của mình thế nào?
5. Đọc README ứng dụng của WebArena. Điều gì hỏng khi bạn nâng cấp một trong các phiên bản ứng dụng đã ghim?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| WebArena | "Benchmark agent web" | 812 nhiệm vụ trên 4 ứng dụng tự host; đánh giá kiểu gym |
| VisualWebArena | "WebArena bản thị giác" | WebArena neo vào thị giác; ảnh chụp màn hình là quan sát |
| OSWorld | "Benchmark agent desktop" | 369 nhiệm vụ trên Ubuntu/Windows/macOS thật |
| GUI grounding | "Ánh xạ pixel sang phần tử" | Model định vị phần tử UI trong 1920x1080 |
| Operational knowledge | "Hiểu biết về OS" | Menu nào, phím tắt nào, ngăn tùy chọn nào |
| OSWorld-G | "Bộ grounding" | 564 mẫu chỉ grounding + bộ huấn luyện |
| OSWorld-Human | "Quỹ đạo vàng" | Chuỗi hành động chuyên gia thủ công để đo hiệu quả |
| Trajectory efficiency | "Số bước so với vàng" | Số bước của agent chia cho mức tối thiểu của con người |

## Đọc thêm

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — benchmark web bốn ứng dụng
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — benchmark desktop đa hệ điều hành
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — năng lực của Claude được định hình theo benchmark
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — con số OSWorld và WebArena
