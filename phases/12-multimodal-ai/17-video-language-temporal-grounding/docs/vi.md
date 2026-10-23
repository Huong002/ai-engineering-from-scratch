# Mô hình video-ngôn ngữ (video-language): token thời gian và định vị theo thời gian

> Video không phải là một chồng ảnh. Một clip 5 giây có thứ tự nhân quả, động từ hành động và thời điểm sự kiện mà mô hình ảnh không thể biểu diễn. Video-LLaMA (Zhang et al., tháng 6/2023) ra mắt video-LLM mở đầu tiên có khả năng định vị âm thanh-hình ảnh. VideoChat và Video-LLaVA mở rộng khuôn mẫu đó. Đến 2025, TMRoPE của Qwen2.5-VL thu hẹp khoảng cách với các mô hình độc quyền tiên tiến. Mỗi hệ thống giải quyết token thời gian (temporal token) theo cách khác nhau — Q-former cho mỗi clip, concat-pool cho mỗi khung hình, TMRoPE cho mỗi token. Bài này đọc các khuôn mẫu, tự viết bộ lấy mẫu khung hình đều-so-với-động, và đánh giá trên các tác vụ định vị theo thời gian (temporal grounding).

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Giai đoạn 12, Bài 08 (LLaVA-OneVision)
**Time:** ~180 phút

## Mục tiêu học tập

- Giải thích vì sao mã hoá vị trí thời gian làm thay đổi hiệu năng VLM video một cách độc lập với bộ mã hoá thị giác.
- So sánh cách lấy mẫu khung hình đều, FPS động và theo sự kiện trên hai trục token-mỗi-giây và độ chính xác định vị.
- Mô tả các thiết kế Q-former-cho-mỗi-clip (Video-LLaMA), pool-cho-mỗi-khung (Video-LLaVA) và M-RoPE-cho-mỗi-token (Qwen2.5-VL).
- Kể tên bốn benchmark video: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## Vấn đề

Một video 1 phút ở 30 FPS (tốc độ khung hình) là 1800 khung hình. Với 196 token thị giác mỗi khung (ViT-B ở 224), đó là 352k token — lớn hơn mọi ngữ cảnh LLM thời 2024.

Có ba chiến lược giảm bớt:

1. Lấy mẫu thưa các khung hình (1-8 FPS tuỳ nội dung).
2. Pool mạnh token patch của từng khung (pool bilinear 3x3 hoặc 4x4).
3. Nén qua Q-former nhận một clip 16 khung và xuất ra 64 token.

Mỗi đánh đổi một khác. Lấy mẫu thưa mất chi tiết thời gian. Pool mất chi tiết không gian. Q-former mất một chút cả hai nhưng tiết kiệm token.

Mã hoá vị trí thời gian là trục còn lại: làm sao mô hình biết khung 5 đến trước khung 6? Các lựa chọn gồm RoPE thời gian 1D đơn giản (Video-LLaMA), embedding thời gian học được (Video-LLaVA), và TMRoPE (Qwen2.5-VL, 3D đầy đủ).

## Khái niệm

### Video-LLaMA: Q-former cho mỗi clip + nhánh âm thanh

Video-LLaMA (2023) là video-LLM mở đầu tiên. Kiến trúc:

- Clip 16 khung ở 2 FPS (tức 8 giây).
- Đặc trưng ViT từng khung -> Video Q-former cross-attend trên cả 16 khung -> 32 query học được -> LLM.
- Nhánh âm thanh song song: waveform -> bộ mã hoá âm thanh ImageBind -> Audio Q-former -> 32 query -> LLM.

Điểm mạnh: suy luận kết hợp âm thanh-hình ảnh. Điểm yếu: độ dài clip cố định, không định vị được mốc thời gian tuỳ ý.

### VideoChat và Video-LLaVA

VideoChat giữ ý tưởng Video-LLaMA nhưng bỏ âm thanh và đơn giản hoá. Video-LLaVA (Lin et al., 2023) huấn luyện một bộ mã hoá thị giác duy nhất trên cả ảnh và khung video ("alignment before projection"), cho một biểu diễn thống nhất. Cả hai đều là bộ mã hoá CLIP đóng băng + MLP + LLM.

Cả hai đều không xử lý được video dài. Đều là hệ thống 8-16 khung.

### Qwen2.5-VL và TMRoPE

Qwen2.5-VL giới thiệu TMRoPE — Temporal-Modality Rotary Position Embedding. Mỗi token patch mang một vị trí (t, h, w), trong đó t là mốc thời gian (timestamp) thực tế chứ không phải chỉ số khung.

Khác biệt chính so với embedding thời gian đơn giản:

- Thời gian tuyệt đối, không phải chỉ số. Mô hình thấy "ở giây 4.2" chứ không phải "ở khung 15."
- Xoay theo từng token, không theo từng clip. Mỗi token thị giác xoay độc lập theo mốc thời gian của nó.
- Tương thích với FPS động. Nếu bạn lấy mẫu 2 FPS chỗ này và 4 FPS chỗ kia, TMRoPE xử lý khoảng cách không đều một cách tự nhiên.

TMRoPE cho phép các truy vấn kiểu "con mèo nhảy ở giây thứ mấy?". Mô hình có thể xuất "ở giây 4.2." Video-LLaMA chỉ có thể nói "đầu clip."

### Các chiến lược lấy mẫu khung hình

Đều: lấy N khung rải đều trên toàn độ dài. Đơn giản, nhưng bỏ lỡ các đỉnh chuyển động.

FPS động: lấy mẫu thích ứng theo cường độ chuyển động. Optical flow hoặc hiệu khung hình chọn ra các đoạn (segment) chuyển động mạnh để lấy mẫu dày hơn. Qwen2.5-VL huấn luyện theo cách này.

Theo sự kiện: chạy một bộ phát hiện nhẹ, lấy mẫu nhiều hơn ở nơi có hành động. Được VideoAgent dùng.

Keyframe + ngữ cảnh: lấy mẫu tại các biên (boundary) cảnh quay cùng vài khung kề. Dùng cho nội dung điện ảnh.

### Pool theo từng khung

Ở 1 FPS và 576 token mỗi khung, một clip 5 phút là 172.800 token. Khả thi với ngữ cảnh 128k của Qwen2.5-VL-72B nhưng tốn kém.

Pool bilinear 3x3 giảm còn 64 token mỗi khung -> 19.200 token cho 5 phút. Điểm cân bằng cho hầu hết tác vụ.

Pool mạnh tay hơn (6x6 -> 16 token mỗi khung) cho các workflow agent, nơi chi tiết không gian ít quan trọng hơn.

### Bốn benchmark video

- VideoMME: hiểu video toàn diện, ngắn + trung + dài.
- TempCompass: suy luận thời gian mịn, câu hỏi "trước" / "sau".
- EgoSchema: video góc nhìn thứ nhất, tầm dài.
- Video-MMMU: câu hỏi video đa ngành, đa phương thức.

Một đánh giá video-VLM đầy đủ chạm tới cả bốn. Chúng nhấn vào các trục khác nhau — TempCompass thiên về thứ tự, EgoSchema về suy luận 3+ phút, VideoMME trải nhiều độ dài.

### Định dạng đầu ra của định vị

Các định dạng đầu ra cho định vị theo thời gian:

- Văn bản tự do: "Con mèo nhảy quanh mốc 4 giây." Dễ phân tích nhưng không chính xác.
- JSON có cấu trúc: `{"event": "jump", "start": 4.1, "end": 4.3}`. Qwen2.5-VL huấn luyện dạng này.
- Dựa trên token: các token đặc biệt `<time>4.1</time>` xen giữa câu trả lời. Định dạng nội bộ của Qwen2.5-VL.

Dựa trên token là chính xác nhất cho việc dùng về sau. Định dạng JSON của Qwen2.5-VL phân tích được trực tiếp.

### Thực hành tốt nhất năm 2026

Cho VLM video năm 2026:

- Bộ mã hoá: SigLIP 2 với M-RoPE hoặc TMRoPE (Qwen2.5-VL).
- Lấy mẫu khung hình: FPS động (1-4 tuỳ chuyển động) kèm giới hạn số khung tối đa.
- Pool mỗi khung: bilinear 3x3.
- Đầu ra: JSON có cấu trúc với trường thời gian + sự kiện.
- Benchmark: VideoMME + TempCompass cho tổng quát; EgoSchema cho tầm dài.

```figure
video-temporal-patches
```

## Áp dụng

`code/main.py` bao gồm:

- Bộ lấy mẫu khung hình đều và FPS động.
- Một bộ đánh giá định vị thời gian đồ chơi: cho một sự kiện "ground truth" tại thời điểm T và đầu ra của mô hình, chấm độ chính xác theo dung sai.
- Một so sánh giữa Video-LLaMA (16 khung, Q-former), Video-LLaVA (8 khung, MLP), Qwen2.5-VL (FPS động + TMRoPE).

## Đóng gói

Bài này tạo ra `outputs/skill-video-vlm-frame-planner.md`. Cho một tác vụ video (giám sát, nhận dạng hành động, định vị theo thời gian, tóm tắt), nó chọn bộ lấy mẫu khung hình, hệ số pool, định dạng đầu ra và mức chính xác kỳ vọng.

## Bài tập

1. Với một demo nấu ăn 3 phút, chọn lấy mẫu đều hay FPS động. Biện luận bằng con số token.

2. TMRoPE thêm đúng cái gì mà một bảng embedding thời gian đơn giản không làm được?

3. Viết một JSON schema cho định vị theo thời gian mà VLM có thể học để xuất ra. Nêu cả các trường hợp lỗi.

4. Đọc Mục 3 của Video-LLaVA về "Alignment Before Projection." Vì sao cách này tốt hơn huấn luyện bộ mã hoá ảnh và video riêng biệt?

5. Dựa trên bảng xếp hạng VideoMME, khoảng cách giữa mô hình mở tốt nhất và mô hình độc quyền tốt nhất tính đến 2026 là bao nhiêu? Bao nhiêu phần của khoảng cách đó đến từ mã hoá thời gian so với quy mô LLM nền?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Temporal grounding | "Câu trả lời định vị theo thời gian" | VLM xuất một khoảng mốc thời gian cụ thể cho thời điểm sự kiện xảy ra |
| TMRoPE | "Time-Multimodal RoPE" | Vị trí xoay 3D với mốc thời gian tuyệt đối, dùng bởi Qwen2.5-VL |
| Dynamic FPS | "Lấy mẫu theo chuyển động" | Lấy mẫu nhiều khung hơn ở các đoạn chuyển động mạnh, ít hơn ở đoạn tĩnh |
| Frame pooling | "Nén không gian mỗi khung" | Giảm số patch mỗi khung bằng nội suy bilinear trước khi vào LLM |
| Video Q-former | "Bộ nén clip" | Nút thắt cross-attention ánh xạ N khung thành K query học được |
| VideoMME | "Bench video" | Benchmark video ngắn/trung/dài toàn diện, 2500+ mẫu |

## Đọc thêm

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)

---

[← Bài trước: MIO và các mô hình đa phương thức any-to-any (bất kỳ sang bất kỳ) truyền trực tiếp](../../16-mio-any-to-any-streaming/docs/vi.md) · [Bài sau: Hiểu long video (video dài) ở cửa sổ ngữ cảnh hàng triệu token →](../../18-long-video-million-token/docs/vi.md)
