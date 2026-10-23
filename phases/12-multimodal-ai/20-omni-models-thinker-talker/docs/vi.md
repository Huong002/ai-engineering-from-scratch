# Omni Models: Qwen2.5-Omni và cách tách Thinker-Talker

> Bản demo sản phẩm GPT-4o tháng 5 năm 2024 gây chấn động không phải vì mô hình bên dưới mà vì hình dạng sản phẩm — một giao diện giọng nói nơi bạn nói, mô hình nhìn thấy những gì camera thấy, và nó đáp lại trong dưới 250ms. Hệ sinh thái mở đã dành phần còn lại của 2024 và 2025 để đua tới bề mặt sản phẩm đó. Qwen2.5-Omni (tháng 3 năm 2025) là thiết kế mở tham chiếu cho omni model (mô hình vạn năng): một Thinker (bộ suy nghĩ — transformer sinh văn bản lớn) cộng với một Talker (bộ nói — transformer sinh giọng nói song song), liên kết bằng token giọng nói streaming (truyền trực tiếp). Mini-Omni đơn giản hoá nó, Moshi bắt kịp latency (độ trễ) của nó, GLM-4-Voice mở rộng nó sang tiếng Trung. Bài này đọc kiến trúc Thinker-Talker và ngân sách độ trễ khiến hội thoại thời gian thực qua streaming trở nên khả thi.

**Type:** Build
**Languages:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Prerequisites:** Giai đoạn 12, Bài 19 (LLM âm thanh), Giai đoạn 12, Bài 16 (any-to-any)
**Time:** ~180 phút

## Mục tiêu học tập

- Tách pipeline suy luận thành Thinker (suy luận văn bản) và Talker (tổng hợp giọng nói) rồi giải thích vì sao streaming song song lại hiệu quả.
- Tính ngân sách thời gian tới byte âm thanh đầu tiên (TTFAB) cho một lượt hội thoại, theo từng thành phần.
- Mô tả cách mã hoá vị trí căn theo thời gian TMRoPE trên hình ảnh, âm thanh và văn bản bên trong Thinker.
- Kể tên ba mẫu hội thoại thời gian thực: half-duplex, turn-taking, full-duplex (song công).

## Vấn đề

Một trợ lý giọng nói thời gian thực phải làm rất nhiều thứ, thật nhanh:

1. Nghe người dùng. Token hoá giọng nói thời gian thực, phát hiện hoạt động giọng nói (VAD) để biết khi nào người dùng nói xong.
2. Tuỳ chọn nhìn. Đầu vào camera ở 2-4 FPS, được truyền vào Thinker song song với âm thanh.
3. Suy nghĩ. Soạn câu trả lời dựa trên lịch sử hội thoại.
4. Nói. Tổng hợp token âm thanh, giải mã thành dạng sóng, truyền tới loa của người dùng.

Mỗi bước đều thêm độ trễ. Cảm giác hội thoại đòi hỏi tổng vòng khứ hồi < 500ms — dưới mức đó, người dùng không còn nhận ra độ trễ. GPT-4o công bố ~250ms. Moshi ~160ms. Qwen2.5-Omni ~350-500ms.

Mọi thành phần đều phải streaming. Không gì được phép "gom hết mọi thứ rồi mới giải mã".

## Khái niệm

### Thinker và Talker

Cách phân rã của Qwen2.5-Omni:

- Thinker: một transformer sinh văn bản 7B-80B. Nhận token văn bản + hình ảnh + âm thanh xen kẽ. Xuất ra token văn bản thể hiện điều cần nói.
- Talker: một transformer sinh giọng nói nhỏ hơn (200M-1B). Nhận token văn bản đầu ra của Thinker cùng các token ngữ cảnh giọng nói gần nhất. Xuất ra token giọng nói rời rạc (chỉ số residual-VQ).
- Speech decoder (bộ giải mã giọng nói): một bộ giải mã dạng sóng streaming (họ SNAC, MoVQGAN) nhận token giọng nói và cho ra mẫu âm thanh trong thời gian thực.

Sự tách biệt này quan trọng. Thinker phải lớn để suy luận tốt. Talker có thể nhỏ vì việc của nó là cục bộ — chuyển văn bản thành token giọng nói. Talker lớn hơn không diễn đạt tốt hơn; nó chỉ chậm hơn.

Chạy cả hai song song:

1. Thinker phát ra token văn bản t_i.
2. Talker nhận t_i (qua streaming) và phát ra token giọng nói s_i, s_{i+1}, ..., s_{i+k}.
3. Bộ giải mã giọng nói nhận token giọng nói ngay khi chúng đến và phát ra mẫu âm thanh.
4. Khi Thinker đang ở token văn bản t_{i+3} thì Talker đã streaming âm thanh cho t_0..t_{i+2} rồi.

### TMRoPE — vị trí đa phương thức căn theo thời gian

Thinker cần tích hợp các khung hình (đến với tốc độ, giả sử, 4 FPS), khung âm thanh (đến với 50 khung/giây), và văn bản từ lịch sử hội thoại. Một thứ tự chuỗi ngây thơ (tất cả hình ảnh, rồi tất cả âm thanh, rồi văn bản) sẽ làm mất căn chỉnh thời gian.

TMRoPE gán dấu thời gian tuyệt đối cho mọi token. Token thị giác ở t=2.3s. Token âm thanh ở t=2.32s. Token văn bản từ người dùng "stop" ở t=2.35s. RoPE xoay attention theo dấu thời gian; mô hình thấy chúng đồng thời về thời gian.

Đây là hạ tầng để câu "anh ấy vẫy tay trong khi nói xin chào" hoạt động — mô hình thấy khung video và âm thanh ở cùng một thời điểm khái niệm.

### Tổng hợp giọng nói dạng streaming

Token giọng nói phải streaming. Mini-Omni (Xie & Wu, 2024) giới thiệu "language models can hear, talk while thinking in streaming": token đầu ra của Thinker và token đầu ra của Talker xen kẽ trong cùng một chuỗi. Talker kích hoạt ngay khi Thinker chốt token văn bản kế tiếp. Không có ranh giới lô.

Moshi (Défossez và cộng sự, tháng 10 năm 2024) là bản triển khai mở nhanh nhất. TTFAB 160ms trên một A100. Kiến trúc: một transformer 7B duy nhất phát ra token văn bản và token giọng nói ở các vị trí xen kẽ, với một "inner monologue" (độc thoại nội tâm) tách luồng suy nghĩ khỏi luồng nói. Đây thực chất là Thinker + Talker hợp nhất vào một mô hình nhờ huấn luyện cẩn thận.

### VAD và turn-taking

Phát hiện hoạt động giọng nói chạy ở phía đầu vào. Hai mẫu:

- Half-duplex: người dùng nói, mô hình nghe. Mô hình nói, người dùng nghe. Bàn giao rõ ràng qua phát hiện im lặng của VAD (~200ms).
- Full-duplex: cả hai có thể nói đồng thời. Mô hình có thể đệm giọng ("uh-huh") hoặc ngắt lời. Khó hơn nhiều. Moshi hỗ trợ việc này.

Qwen2.5-Omni mặc định hỗ trợ half-duplex, với turn-taking qua ngưỡng khoảng lặng. Full-duplex cần xử lý ở tầng ứng dụng.

### Qwen3-Omni (tháng 11 năm 2025)

Bản kế nhiệm. Thinker Qwen3-80B, Talker lớn hơn, TMRoPE-v2 cải tiến. Độ trễ gần với 250ms của GPT-4o. Trọng số mở. Điểm benchmark trên OmniBench cạnh tranh với Gemini 2.0 Live.

### Ngân sách độ trễ khi chạy thật

Với một lượt tương tác streaming điển hình:

- Mic -> token âm thanh: 40-80ms.
- Prefill (prompt + lịch sử): 100-200ms ở 7B, nhiều hơn hẳn ở 70B.
- Token văn bản đầu tiên của Thinker: 40ms.
- Talker xử lý token văn bản đầu tiên: 20ms.
- Các token giọng nói đầu tiên được chốt: 40ms.
- Giải mã residual-VQ: 30ms.
- Giải mã dạng sóng giọng nói: 50-80ms.

Tổng TTFAB: 320-510ms ở 7B, 600-900ms ở 70B. Chất lượng hàng đầu thường cần 70B+; do đó có khoảng cách độ trễ so với mức tốt nhất.

### Phép tính tốc độ token

Với giọng nói 16kHz và 50 Hz token giọng nói lớp nền, bạn cần 50 token giọng nói cho mỗi giây đầu ra. Talker phải phát ≥50 tok/s để theo kịp. Với thông lượng LLM điển hình 30-80 tok/s trên H100, một Talker nhỏ (200-300M) là đủ nhanh; Talker 7B sẽ tụt lại.

Đó là lý do tồn tại các mô hình Talker chuyên dụng nhỏ, thay vì "cứ dùng mô hình chính".

```figure
l5-thinker-talker
```

## Áp dụng

`code/main.py`:

- Mô phỏng pipeline Thinker-Talker với tốc độ phát token giả lập.
- Tính TTFAB cho các kích thước mô hình và tần số lấy mẫu mic tuỳ chỉnh.
- Minh hoạ turn-taking half-duplex với ngưỡng im lặng của VAD.

## Đóng gói

Bài này tạo ra `outputs/skill-omni-streaming-budget.md`. Cho trước TTFAB mục tiêu và bộ tính năng của một sản phẩm giọng nói thời gian thực (nhìn vào, song ngữ, full-duplex), chọn Qwen2.5-Omni, Qwen3-Omni, Moshi, hoặc Mini-Omni và định cỡ Thinker/Talker.

## Bài tập

1. TTFAB mục tiêu của bạn là 300ms. Trên Thinker 7B và Talker 300M, hãy viết ra độ trễ của từng thành phần.

2. Qwen2.5-Omni dùng TMRoPE. Mô tả những gì mô hình thấy với một prompt trong đó người dùng bắt đầu nói ở t=1s và camera bắt được một cử chỉ ở t=1.2s.

3. Hỗ trợ full-duplex đòi hỏi mô hình phát âm thanh trong khi vẫn nghe. Đề xuất một định dạng dữ liệu huấn luyện dạy được điều này.

4. Đọc Mục 4 trong bài báo của Moshi. Mô tả cách tách "inner monologue" và vì sao nó tránh được việc phải tách Thinker-Talker.

5. Tính ngân sách thông lượng: một Talker phải phát token nhanh bao nhiêu để theo kịp giọng nói 16kHz với 50 token lớp nền/giây?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Thinker | "Bộ não suy luận" | Transformer sinh văn bản lớn, tạo ra điều cần nói |
| Talker | "Cái miệng sinh giọng nói" | Transformer nhỏ tạo token giọng nói rời rạc từ văn bản của Thinker |
| TTFAB | "Ngân sách độ trễ" | Time-to-first-audio-byte: từ lúc người dùng nói xong tới mẫu âm thanh đầu tiên phát ra |
| TMRoPE | "RoPE căn theo thời gian" | Mã hoá vị trí dùng dấu thời gian tuyệt đối trên hình ảnh, âm thanh, văn bản |
| Half-duplex | "Luân phiên lượt" | Người dùng và mô hình thay phiên nhau; VAD phát hiện khoảng lặng để biết người dùng nói xong |
| Full-duplex | "Đồng thời" | Mô hình có thể nói và nghe cùng lúc; có khả năng đệm giọng |
| Inner monologue | "Cách tách của Moshi" | Thiết kế một mô hình duy nhất, nơi luồng suy nghĩ và luồng nói xen kẽ |

## Đọc thêm

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)

---

[← Bài trước: Mô hình âm thanh-ngôn ngữ: cung đường từ Whisper tới Audio Flamingo 3](../../19-audio-language-whisper-to-af3/docs/vi.md) · [Bài sau: VLA có thân thể: RT-2, OpenVLA, π0, GR00T →](../../21-embodied-vlas-openvla-pi0-groot/docs/vi.md)
