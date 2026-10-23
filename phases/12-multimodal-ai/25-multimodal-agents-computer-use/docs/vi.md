# Agent đa phương thức và computer use (dùng máy tính) — Capstone

> Sản phẩm tiên phong của năm 2026 là một agent đa phương thức biết đọc ảnh chụp màn hình (screenshot), bấm nút, điều hướng giao diện web, điền biểu mẫu và hoàn tất quy trình từ đầu đến cuối. SeeClick và CogAgent (2024) đã chứng minh nguyên thuỷ định vị GUI (grounding). Ferret-UI bổ sung cho di động. ChartAgent đưa vào cách dùng công cụ thị giác cho biểu đồ. VisualWebArena và AgentVista (2026) là những bộ đánh giá mà giới tiên phong đang theo đuổi — và ngay cả Gemini 3 Pro lẫn Claude Opus 4.7 cũng chỉ đạt ~30% trên các bài khó của AgentVista. Bài tổng kết này nối lại mọi sợi chỉ của Giai đoạn 12: tri giác (VLM độ phân giải cao), lập luận (LLM có dùng công cụ), định vị (xuất ra toạ độ — coordinate), bộ nhớ tầm xa và đánh giá.

**Type:** Capstone
**Languages:** Python (stdlib, action schema + agent loop skeleton)
**Prerequisites:** Giai đoạn 12, Bài 05 (LLaVA), Giai đoạn 12, Bài 09 (Qwen-VL JSON), Giai đoạn 14 (Kỹ nghệ agent)
**Time:** ~240 phút

## Mục tiêu học tập

- Thiết kế một vòng lặp agent (agent loop): tri giác → lập luận → hành động → quan sát → lặp lại.
- Xây dựng một lược đồ đầu ra định vị GUI (click toạ độ, gõ văn bản, cuộn, kéo) mà VLM có thể phát ra dưới dạng JSON.
- So sánh agent chỉ dùng ảnh chụp màn hình với agent dùng cây trợ năng (accessibility tree) và agent lai.
- Dựng một đánh giá benchmark agent đa phương thức trên một lát cắt nhỏ của VisualWebArena.

## Vấn đề

Một quy trình trên trang đặt vé: "tìm cho tôi chuyến bay tới Tokyo ngày 15 tháng 4, ghế cạnh lối đi dưới 800 đô, đặt luôn."

Agent đa phương thức cần:

1. Chụp ảnh chụp màn hình của trình duyệt.
2. Phân tích ảnh chụp màn hình + URL + mục tiêu thành một kế hoạch.
3. Phát ra một hành động có cấu trúc: click (tại x,y), gõ "Tokyo" (tại phần tử E), cuộn xuống, chọn (nút radio).
4. Áp dụng hành động lên trình duyệt.
5. Quan sát trạng thái mới (ảnh chụp màn hình tiếp theo).
6. Lặp lại cho tới khi xong việc.

Mỗi bước là một lần gọi VLM đa phương thức. Đầu ra của VLM phải là JSON phân tích được. Sai sót tích luỹ qua các bước, nên khả năng phục hồi rất quan trọng.

## Khái niệm

### GUI grounding — nguyên thuỷ cơ bản

Định vị GUI nghĩa là: cho một ảnh chụp màn hình và một chỉ dẫn bằng ngôn ngữ tự nhiên, xuất ra toạ độ (x, y) để click (hoặc hành động khác).

SeeClick (arXiv:2401.10935) là kết quả mở đầu tiên ở quy mô lớn: fine-tune một VLM trên dữ liệu GUI tổng hợp + thật, xuất toạ độ dưới dạng token văn bản thuần. Hiệu quả.

CogAgent (arXiv:2312.08914) thêm mã hoá độ phân giải cao 1120x1120 cho các UI dày đặc. Điểm: ~84% trên điều hướng web.

Ferret-UI (arXiv:2404.05719) tập trung vào UI di động, tích hợp với dữ liệu trợ năng của iOS.

Định dạng đầu ra thường là JSON:

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` giúp phục hồi: nếu toạ độ lệch giữa các ảnh chụp màn hình, gợi ý ngữ nghĩa cho hệ thống định vị lại.

### Lược đồ hành động

Một lược đồ hành động điển hình có 6-10 loại hành động:

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

Agent phát ra một hành động mỗi bước. Lớp bọc trình duyệt thực thi và trả về trạng thái mới.

### Chỉ ảnh chụp màn hình so với cây trợ năng

Hai chế độ đầu vào:

- Chỉ ảnh chụp màn hình: ảnh đầy đủ, không có thông tin cấu trúc. Tổng quát nhất; chạy được trên mọi ứng dụng.
- Cây trợ năng: DOM có cấu trúc / thông tin trợ năng của iOS. Định vị đáng tin hơn nhiều; chạy được ở đâu có cây này.
- Lai: cả hai, dùng cây làm bộ định vị đáng tin cho hành động nguyên tử và ảnh chụp màn hình cho ngữ cảnh ngữ nghĩa.

Agent trong sản phẩm dùng kiểu lai khi có thể. Tự động hoá trình duyệt (Selenium + trợ năng) luôn có cây; ứng dụng desktop đôi khi có.

### Bộ nhớ dài hạn

Một quy trình 20 bước sinh ra 20 ảnh chụp màn hình. Ngữ cảnh của VLM đầy rất nhanh. Ba chiến lược nén:

- Chuỗi tóm tắt: cứ sau 5 bước, tóm tắt những gì đã xảy ra, bỏ các ảnh chụp màn hình cũ.
- Bỏ khung hình: giữ ảnh đầu, ảnh cuối và mỗi ảnh thứ 3.
- Nhật ký do công cụ ghi: thực thi hành động, giữ một nhật ký văn bản về những gì đã làm; không xem lại ảnh chụp màn hình cũ.

API computer use của Claude dùng mẫu nhật ký. Đơn giản hơn, đáng tin hơn.

### Dùng công cụ thị giác

ChartAgent (arXiv:2510.04514) giới thiệu việc dùng công cụ thị giác để hiểu biểu đồ: cắt, phóng to, OCR, gọi bộ phát hiện bên ngoài. Agent có thể xuất "cắt vùng (100, 200, 300, 400) rồi gọi OCR" như một lệnh gọi công cụ. Công cụ trả về văn bản; VLM tiếp tục lập luận.

Mẫu này khái quát hoá: set-of-mark prompting, chú thích vùng và công cụ phát hiện bên ngoài đều khớp cùng một lược đồ "xuất một lệnh gọi công cụ, nhận một phản hồi có cấu trúc".

### Các benchmark của năm 2026

- ScreenSpot-Pro. Định vị GUI trên ~1k ảnh chụp màn hình web. SOTA mở Qwen2.5-VL-72B ~85%. Tiên phong ~90%.
- VisualWebArena. Tác vụ web đầu-cuối (mua sắm, diễn đàn, rao vặt). SOTA mở ~20%. Gemini 3 Pro ~27%.
- AgentVista (arXiv:2602.23166). Benchmark khó nhất của năm 2026. Quy trình thực tế trên 12 lĩnh vực. Mô hình tiên phong đạt 27-40%; mô hình mở 10-20%.
- WebArena / WebShop. Benchmark cũ hơn; đã bị giới tiên phong chạm trần.

### Vì sao vẫn còn khó

Các nút thắt hiệu năng của agent:

1. Định vị thị giác ở quy mô nhỏ. "Bấm vào dấu X nhỏ" hay thất bại ở độ phân giải di động.
2. Lập kế hoạch tầm xa. Sau 10 hành động, agent trôi khỏi mục tiêu.
3. Phục hồi lỗi. Khi một cú click thất bại (sai nút), phát hiện + phục hồi hiếm khi có trong dữ liệu huấn luyện.
4. Ngữ cảnh xuyên trang. Nhảy giữa các tab hoặc biểu mẫu dài làm mất trạng thái.

Hướng nghiên cứu: kiến trúc bộ nhớ, lập kế hoạch lại tường minh, kiểm chứng đa phương thức (khớp ảnh chụp màn hình để biết hành động thành công).

### Bài tổng kết tự tay làm

Nhiệm vụ của bài tổng kết: xây một agent computer use mà:

1. Đọc HTML + ảnh chụp màn hình của một trang giả lập đặt vé.
2. Lập kế hoạch chuỗi nhiều bước: tìm kiếm → chọn → điền biểu mẫu → gửi.
3. Phát ra các hành động JSON khớp lược đồ hành động.
4. Đánh giá trên một lát cắt cố định gồm 10 tác vụ.

Bài học cung cấp mã khung dễ mở rộng thành trình duyệt thật.

```figure
mm-agent-loop
```

## Áp dụng

`code/main.py` là mã khung của bài tổng kết:

- Định nghĩa JSON lược đồ hành động (10 hành động).
- Trạng thái trình duyệt giả lập dưới dạng dict.
- Khung vòng lặp agent: nhận trạng thái, phát hành động, áp dụng, lặp.
- Mini-benchmark 10 tác vụ (trang tổng hợp) để đo tỉ lệ thành công đầu-cuối.
- Hook phục hồi lỗi cho khi một hành động thất bại.

## Đóng gói

Bài này tạo ra `outputs/skill-multimodal-agent-designer.md`. Cho một sản phẩm computer use (lĩnh vực, bộ hành động, mục tiêu đánh giá), thiết kế toàn bộ vòng lặp agent, chiến lược bộ nhớ, chế độ định vị và điểm benchmark kỳ vọng.

## Bài tập

1. Mở rộng lược đồ hành động với một công cụ `screenshot_region` (cắt + phóng to). Những tác vụ nào hưởng lợi?

2. Đọc AgentVista (arXiv:2602.23166). Mô tả hạng mục tác vụ khó nhất và vì sao các mô hình tiên phong vẫn thất bại.

3. Nén bộ nhớ tầm xa: thiết kế một chuỗi tóm tắt giữ ≤4 ảnh chụp màn hình còn sống, số lượng bất kỳ được ghi log.

4. Xây một hook phục hồi lỗi: khi hành động thất bại (không tìm thấy nút), agent làm gì tiếp theo?

5. So sánh Claude 4.7 chỉ dùng ảnh chụp màn hình với Qwen2.5-VL lai ảnh chụp màn hình + cây trợ năng trên 10 tác vụ web. Bên nào thắng ở tác vụ nào?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| GUI grounding | "Click toạ độ" | Mô hình xuất ra (x,y) cho mục tiêu của một chỉ dẫn trên ảnh chụp màn hình |
| Action schema | "Định nghĩa công cụ" | Mô tả JSON của các hành động hợp lệ (click, type, scroll, drag) |
| Accessibility tree | "DOM có cấu trúc" | Cây phân cấp UI đọc được bằng máy từ API trình duyệt/iOS |
| Hybrid agent | "Ảnh chụp màn hình + cây" | Dùng cả ảnh và thông tin cấu trúc; đáng tin hơn từng thứ riêng lẻ |
| Visual tool use | "Zoom/crop/detect" | Agent gọi công cụ thị giác bên ngoài (OCR, phát hiện) giữa kế hoạch |
| Summary-chain | "Nén bộ nhớ" | Tóm tắt văn bản định kỳ thay cho lịch sử ảnh chụp màn hình dài |
| VisualWebArena | "Bench web đầu-cuối" | Benchmark 2024 cho tác vụ web đầu-cuối |
| AgentVista | "Bench khó 2026" | Quy trình thực tế 12 lĩnh vực; ngay Gemini 3 Pro cũng chỉ đạt ~30% |

## Đọc thêm

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)

---

[← Bài trước: RAG đa phương thức và Truy hồi liên phương thức](../../24-multimodal-rag-cross-modal/docs/vi.md)
