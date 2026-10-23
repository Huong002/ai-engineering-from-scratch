# Họ Qwen-VL và Video Dynamic-FPS

> Họ Qwen-VL — Qwen-VL (2023), Qwen2-VL (2024), Qwen2.5-VL (2025), Qwen3-VL (2025) — là dòng mô hình ngôn ngữ-thị giác mở có ảnh hưởng lớn nhất năm 2026. Mỗi thế hệ đặt một cược kiến trúc quyết định mà phần còn lại của hệ sinh thái mở sao chép trong vòng mười hai tháng: resolution (độ phân giải) động gốc qua M-RoPE, lấy mẫu dynamic fps (khung hình/giây động) với căn chỉnh thời gian tuyệt đối, window attention trong ViT, và định dạng đầu ra agent có cấu trúc. Đến Qwen3-VL, công thức đã ổn định: một vision encoder (bộ mã hoá thị giác) 2D-RoPE-ViT với đầu vào giữ tỉ lệ khung hình gốc, một projector MLP nối vào nền ngôn ngữ Qwen3 lớn, và các giai đoạn huấn luyện nhấn mạnh OCR, grounding (định vị đối tượng) và hành vi agent như mục tiêu hạng nhất. Bài này đọc dòng họ này theo trình tự thời gian để bạn hiểu vì sao mỗi núm vặn nằm ở chỗ nó nằm.

**Type:** Learn
**Languages:** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)
**Prerequisites:** Giai đoạn 12, Bài 06 (patch-n'-pack)
**Time:** ~120 phút

## Mục tiêu học tập

- Tính ba phép quay của M-RoPE theo ba trục — temporal (theo thời gian), chiều cao, chiều rộng — và giải thích vì sao cần cả ba.
- Chọn một chiến lược lấy mẫu dynamic fps cho một video và lý giải về token-mỗi-giây so với độ chính xác phát hiện sự kiện.
- Kể tên bốn nâng cấp qua các thế hệ Qwen-VL theo đúng thứ tự và mỗi nâng cấp mở ra điều gì.
- Nối một định dạng đầu ra agent kiểu JSON của Qwen2.5-VL và phân tích các lệnh gọi công cụ có cấu trúc từ phản hồi của VLM.

## Vấn đề

Qwen-VL ra mắt tháng 8 năm 2023 như phản ứng trực tiếp với LLaVA-1.5 và BLIP-2. Khoảng trống mà nhóm Qwen nhắm tới gồm ba phần: độ phân giải, video và đầu ra có cấu trúc.

Độ phân giải: LLaVA-1.5 chạy ở 336x336. Đủ tốt cho ảnh chụp, vô dụng với một hoá đơn tiếng Trung hay ảnh chụp bảng tính dày đặc. Đổi mới đầu tiên của Qwen-VL là 448x448 và đầu ra bounding box (khung bao) có định vị, cho phép mô hình chỉ tay vào sự vật.

Video: Video-LLaMA xếp chồng các encoder theo từng khung hình rồi đưa chúng vào LLM. Cách đó hiệu quả với clip ngắn, không hiệu quả với video dài nhiều phút, nơi trục thời gian chính là tín hiệu. Nhóm Qwen muốn một encoder duy nhất hiểu được thời gian.

Đầu ra có cấu trúc: LLaVA phát ra văn bản tự do. Một agent cần JSON. Qwen-VL được huấn luyện trên các định dạng đầu ra JSON tường minh, kể cả toạ độ khung bao dưới dạng văn bản.

Mỗi thế hệ Qwen-VL mở rộng một trong ba trục này.

## Khái niệm

### Qwen-VL (tháng 8 năm 2023)

Thế hệ đầu tiên: OpenCLIP ViT-bigG/14 làm encoder (2.5B tham số), Q-Former tương thích LLama (1 bước với 256 query), nền Qwen-7B. Các đóng góp:

- Độ phân giải 448x448 (khi đó là SOTA cho một VLM mở).
- Grounding: huấn luyện trên các cặp ảnh-văn bản với đầu ra token toạ độ tường minh. "Con mèo ở <box>(112, 204), (280, 344)</box>".
- Huấn luyện đa ngữ Trung + Anh ngay từ đầu.

Điểm benchmark khi đó: ngang ngửa GPT-4V ở tiếng Anh, vượt trội ở tiếng Trung. Giám sát grounding mới là điểm nhấn thực sự.

### Qwen2-VL (tháng 9 năm 2024) — M-RoPE và độ phân giải gốc

Qwen2-VL thay bộ kết hợp độ phân giải cố định + Q-Former bằng một ViT encoder có độ phân giải động gốc. Những thay đổi chính:

- Độ phân giải động gốc. ViT nhận mọi kích thước HxW chia hết cho 28 (patch 14 với hợp nhất không gian 2x). Một ảnh 1120x672 (40x24 patch đã hợp nhất) tạo ra 960 token thị giác. Không resize, không chia ô, không thumbnail.
- M-RoPE (Multimodal RoPE). Mỗi token mang vị trí 3D (t, h, w) thay vì 1D. Với ảnh t=0, với video t = frame_index. RoPE quay các vector query/key theo một tần số cho mỗi trục. Không cần bảng positional embedding.
- Projector MLP. Bỏ Q-Former; dùng MLP 2 lớp trên các token patch đã hợp nhất.
- Video với dynamic fps. Video được lấy mẫu ở 1-2 FPS theo mặc định, nhưng mô hình nhận số khung hình tuỳ ý.

Kết quả: Qwen2-VL-7B ngang GPT-4o trên vài benchmark đa phương thức và vượt ở DocVQA (94.5 so với 88.4). Thay đổi kiến trúc chính là nước đi quyết định.

### Qwen2.5-VL (tháng 2 năm 2025) — dynamic fps + thời gian tuyệt đối

Bước chuyển lớn của Qwen2.5-VL là video. Dynamic fps không chỉ là "lấy thêm khung hình khi cần". Bài báo đã hình thức hoá:

- Token thời gian tuyệt đối. Thay vì chỉ số vị trí (khung 0, 1, 2...), dùng dấu thời gian thực. "Ở 0:04, con mèo nhảy." Mô hình thấy các token `<time>0.04</time>` xen giữa các token khung hình.
- Dynamic fps. Lấy mẫu 1 FPS cho cảnh chậm, 4+ FPS cho cảnh hành động. Người dùng hoặc bên huấn luyện chọn; M-RoPE thích ứng.
- Window attention trong ViT. Attention không gian được chia cửa sổ (cục bộ trong từng khối) để tăng thông lượng; attention toàn cục vài lớp một lần.
- Định dạng đầu ra JSON tường minh. Huấn luyện trên dữ liệu gọi công cụ: "{\"tool\": \"click\", \"coords\": [380, 220]}". Sẵn sàng cho agent ngay khi lấy ra.
- Mở rộng MRoPE-v2. Vị trí co giãn theo kích thước đầu vào tối đa để video 10 phút không hết dải tần số.

Benchmark: Qwen2.5-VL-72B vượt GPT-4o trên phần lớn benchmark video, ngang Gemini 2.0 trên tài liệu, và lập SOTA cho mô hình mở ở GUI grounding (ScreenSpot: độ chính xác 84% so với 38% của GPT-4o).

### Qwen3-VL (tháng 11 năm 2025)

Qwen3-VL là bản nâng cấp tăng dần, củng cố thay vì phát minh lại: nền LLM lớn hơn (Qwen3-72B), dữ liệu huấn luyện mở rộng, OCR tốt hơn, suy luận mạnh hơn nhờ "thinking mode" của Qwen3. ViT và M-RoPE giữ nguyên. Bài báo tập trung vào cải thiện dữ liệu và huấn luyện hơn là kiến trúc.

Điều rút ra cho dòng họ này: đến 2025, kiến trúc Qwen-VL đã ổn định. Các thế hệ sau chỉ mở rộng compute và dữ liệu, không thay đổi primitive.

### M-RoPE dưới góc toán học

RoPE cổ điển quay một query `q` có số chiều `d` theo vị trí `m` bằng các cặp toạ độ:

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE chia chiều ẩn thành ba dải. Giả sử `d = 96`. Gán 32 chiều cho trục thời gian, 32 cho chiều cao, 32 cho chiều rộng. Mỗi dải quay theo vị trí trục của nó. Một patch tại (t=5, h=10, w=20) nhận các phép quay `R_t(5)`, `R_h(10)`, `R_w(20)` áp lên ba dải của nó.

Token văn bản dùng `t = text_index, h = 0, w = 0` (hoặc một lựa chọn đã chuẩn hoá) để giữ tương thích. Khung hình video dùng `t = frame_time, h = row, w = col`. Ảnh đơn dùng `t = 0`.

Lợi ích: một kiểu mã hoá vị trí xử lý được văn bản, ảnh và video mà không cần rẽ nhánh code hay bảng vị trí khác nhau.

### Logic lấy mẫu dynamic fps

Cho một video dài `T` giây và một ngân sách token mục tiêu `B`:

1. Tính FPS tối đa bạn có thể chi: `fps_max = B / (T * tokens_per_frame)`.
2. Chọn FPS mục tiêu trong `{1, 2, 4, 8}` thoả `fps <= fps_max`.
3. Nếu chuyển động mạnh (heuristic optical-flow hoặc yêu cầu rõ ràng từ người dùng), chọn FPS cao hơn. Nếu chuyển động yếu, chọn thấp hơn.
4. Lấy mẫu đều ở FPS đã chọn; chèn các token `<time>t</time>` giữa các khung hình.

Qwen2.5-VL huấn luyện logic này một cách ngầm; khi suy luận, người dùng điều khiển qua tham số `fps`. Một chuỗi hành động 60 giây ở 4 FPS với 81 token mỗi khung hình = 19440 token, vẫn vừa trong context 32k.

### Đầu ra agent có cấu trúc

Huấn luyện agent của Qwen2.5-VL nhắm tường minh vào các lệnh gọi công cụ có cấu trúc:

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

Việc phân tích là tất định: JSON.parse trên đầu ra của mô hình. So với dạng tự do "click at (1024, 512)" phải dùng regex và xử lý nhoè nghĩa. Chính sự thay đổi này khiến điểm ScreenSpot của Qwen2.5-VL nhảy từ 55% của Qwen2-VL lên 84%.

```figure
mm-mrope-axes
```

## Áp dụng

`code/main.py` hiện thực:

- Tính vị trí M-RoPE cho một chuỗi đóng gói trộn văn bản, patch ảnh và khung hình video.
- Bộ lấy mẫu dynamic fps: cho (duration, budget, motion_level), chọn FPS và phát ra dấu thời gian các khung hình.
- Một bộ phân tích đầu ra JSON kiểu Qwen2.5-VL đồ chơi, xử lý phản hồi gọi công cụ có trường toạ độ.

Hãy chạy thử, rồi cảm nhận khác biệt khi bạn đổi FPS cố định sang dynamic fps trên một video 5 phút.

## Đóng gói

Bài này tạo ra `outputs/skill-qwen-vl-pipeline-designer.md`. Cho một tác vụ video (giám sát, agent, nhận dạng hành động, khả năng tiếp cận), nó phát ra cấu hình Qwen2.5-VL (ngân sách khung hình, chiến lược FPS, cờ window-attention, chế độ đầu ra agent) và một ước lượng độ trễ. Dùng nó mỗi khi bạn triển khai một mô hình thuộc họ Qwen-VL cho một sản phẩm video.

## Bài tập

1. Tính các phép quay M-RoPE cho một patch tại (t=3, h=5, w=7) với chiều ẩn 48 (16 mỗi dải, base theta 10000). Chỉ ra góc quay của ba cặp đầu tiên trong mỗi dải.

2. Một bản ghi camera an ninh 10 phút ở 1 FPS tạo ra bao nhiêu khung hình? Với độ phân giải 384 và pool 3x, tổng cộng bao nhiêu token? Context 32k mặc định của Qwen2.5-VL có xử lý nổi không?

3. Chọn FPS cho một pha bóng tennis 30 giây, một video demo công thức nấu ăn 30 giây và một bản ghi UI-agent 30 giây. Biện luận từng trường hợp bằng logic dynamic fps.

4. Qwen2.5-VL bỏ hẳn Q-Former. Vì sao một MLP đơn giản hoạt động được vào 2025 mà không được vào 2023? (Gợi ý: quy mô dữ liệu và chất lượng encoder.)

5. Phân tích ba đầu ra gọi công cụ JSON của Qwen2.5-VL thành các dict Python. Điều gì hỏng với JSON sai định dạng và cookbook của Qwen khuyến nghị chiến lược phục hồi nào?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| M-RoPE | "Multimodal RoPE" | Embedding vị trí quay 3D với các dải thời gian, chiều cao và chiều rộng trong chiều ẩn |
| Dynamic FPS | "Lấy mẫu thông minh" | Tốc độ frame sampling (lấy mẫu khung hình) được chọn cho từng video dựa trên chuyển động, thời lượng và ngân sách token |
| Absolute time token | "Token dấu thời gian" | `<time>t</time>` xen trong chuỗi để mô hình thấy số giây thực chứ không phải chỉ số khung |
| Window attention | "Attention cục bộ" | Self-attention không gian giới hạn trong các cửa sổ nhỏ để nhanh hơn; attention toàn cục được thêm định kỳ |
| Structured agent output | "Chế độ JSON" | Dữ liệu huấn luyện dạy VLM phát ra JSON phân tích được, có toạ độ và tên công cụ |
| min_pixels / max_pixels | "Giới hạn độ phân giải" | Các điều khiển của Qwen2.5-VL theo từng request, giới hạn tổng số pixel và do đó giới hạn số token |
| Grounding | "Chỉ vào chỗ đó" | Xuất toạ độ khung bao dưới dạng token văn bản; dùng từ Qwen-VL v1 |

## Đọc thêm

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)

---

[← Bài trước: LLaVA-OneVision: Một ảnh, nhiều ảnh và video trong cùng một mô hình](../../08-llava-onevision-single-multi-video/docs/vi.md) · [Bài sau: InternVL3: Tiền huấn luyện đa phương thức nguyên bản →](../../10-internvl3-native-multimodal/docs/vi.md)
