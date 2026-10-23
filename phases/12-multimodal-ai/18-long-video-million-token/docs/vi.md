# Hiểu long video (video dài) ở cửa sổ ngữ cảnh hàng triệu token

> Một video 4K dài 1 giờ ở 24 FPS, sau khi chia patch và nhúng, sinh ra cỡ 60 triệu token. Một tập podcast 2 giờ được bóc băng là 30.000 token. Một phim Blu-ray trọn vẹn, kể cả khi nén bằng pooling mạnh, là hàng trăm nghìn token. Gemini 1.5 của Google (tháng 3/2024) mở ra kỷ nguyên này với ngữ cảnh 10 triệu token, truy hồi needle-in-a-haystack đáng tin cậy trên những video dài hàng giờ. LWM (Liu và cộng sự, tháng 2/2024) cho thấy đường mở rộng của ring attention. LongVILA và Video-XL mở rộng khả năng nạp vào hơn nữa. VideoAgent đổi ngữ cảnh thô lấy agentic retrieval (truy hồi tác tử). Mỗi cách là một đánh đổi khác nhau về tính toán, truy hồi và độ phức tạp kỹ thuật. Bài này đọc chúng cạnh nhau.

**Type:** Build
**Languages:** Python (stdlib, needle-in-haystack simulator + agentic-retrieval router)
**Prerequisites:** Giai đoạn 12, Bài 17 (token thời gian cho video)
**Time:** ~180 phút

## Mục tiêu học tập

- Tính tổng số token thị giác cho video dài ở các mức FPS và pooling khác nhau.
- Giải thích ba đường mở rộng: ngữ cảnh thô (Gemini 1.5), ring attention (LWM), token compression (nén token) (LongVILA / Video-XL).
- So sánh VLM video dùng ngữ cảnh thô với VLM video dùng agentic retrieval (VideoAgent) về độ chính xác và độ trễ.
- Thiết kế một bài kiểm tra needle-in-a-haystack cho video 30 phút và đo truy hồi tại một phút cụ thể.

## Vấn đề

Một khung hình đơn với patch cỡ Qwen2.5-VL ở độ phân giải gốc 384 là ~729 token. Với pooling 3x3 thì còn 81 token mỗi khung. Một clip 30 phút ở 1 FPS = 1800 khung = 145.800 token. Các VLM mở năm 2025 làm được, nhưng sát nút. Ở 2 FPS, 291.600 token — chỉ những context window (cửa sổ ngữ cảnh) lớn nhất mới chứa nổi.

Một phim 2 giờ ở 1 FPS là 583k token. Vượt quá phần lớn model mở năm 2026; cần Gemini 2.5 Pro hoặc pooling mạnh tay hơn.

Ba đường mở rộng đã xuất hiện.

## Khái niệm

### Đường 1: Ngữ cảnh thô (Gemini 1.5, Claude Opus)

Ném phần cứng vào vấn đề. Mở ngữ cảnh lên hàng triệu token, xử lý mọi thứ trong một lượt forward.

Gemini 1.5 Pro ra mắt với 1M token; Gemini 1.5 Ultra lên 10M; Gemini 2.5 Pro năm 2026 xử lý video hàng giờ một cách đáng tin cậy. Bài báo (arXiv:2403.05530) ghi nhận truy hồi needle-in-a-haystack đạt 99,7% tới ~9,5M token.

Kỹ thuật: một cài đặt attention tùy biến với hệ thống memory (bộ nhớ) phân cấp (local + global + sparse) cộng định tuyến chuyên gia MoE để tăng hiệu quả ở ngữ cảnh dài. Không công bố đầy đủ chi tiết. Không mã nguồn mở.

### Đường 2: Ring attention (LWM, LongVILA)

Ring attention chia chuỗi dài ra nhiều thiết bị theo một "vòng", trong đó mỗi thiết bị giữ một đoạn. Attention trên toàn chuỗi diễn ra bằng cách mỗi thiết bị gửi đoạn của mình cho thiết bị kế tiếp theo kiểu vòng, tính attention từng phần rồi tổng hợp lại.

LWM (Liu và cộng sự, 2024) huấn luyện một model ngữ cảnh 1M token theo cách này. Chi phí tính toán huấn luyện tăng tuyến tính theo ngữ cảnh, không phải bậc hai — cú đánh bậc hai lên attention được san đều qua các thiết bị trong vòng.

LongVILA (arXiv:2408.10188) áp dụng cách này cho VLM. Video 1400 khung với 192 token mỗi khung = 268k ngữ cảnh, huấn luyện bằng ring attention trên song song 8 chiều.

### Đường 3: Token compression (Video-XL, LongVA)

Rẻ hơn ngữ cảnh thô: nén mạnh tay trước khi LLM thấy chuỗi.

Video-XL (arXiv:2409.14485) dùng một token tóm tắt thị giác: mỗi clip N khung sinh ra một token "tóm tắt" duy nhất, attend trên N khung đó. Khi suy luận, LLM thấy một token tóm tắt mỗi clip, thu nhỏ ngữ cảnh đi rất nhiều.

LongVA mở rộng ngữ cảnh LLM từ 200k lên 2M bằng kỹ thuật "long context transfer". Huấn luyện trên văn bản ngữ cảnh dài, rồi chuyển sang video ngữ cảnh dài nhờ biểu diễn chia sẻ.

Token compression đánh đổi truy hồi ở các mốc thời gian cụ thể để lấy khả năng mở rộng. Model biết đại khái chuyện gì đã xảy ra nhưng đôi khi bỏ sót đúng khung hình.

### Đường 4: Agentic retrieval (VideoAgent)

Đừng nạp cả video vào LLM. Thay vào đó, coi video như một cơ sở dữ liệu và dùng LLM để truy vấn nó.

VideoAgent (arXiv:2403.10517):

1. LLM đọc câu hỏi.
2. LLM hỏi một công cụ truy hồi để lấy các clip liên quan ("cho tôi xem các đoạn có con mèo").
3. Công cụ trả về mốc thời gian của các clip khớp.
4. LLM đọc những clip đó qua một VLM.
5. LLM ghép câu trả lời hoặc hỏi truy vấn tiếp.

Đây là mẫu LLM-làm-tác-tử áp vào video dài. Suy luận rẻ hơn (chỉ mã hoá các clip liên quan), kỹ thuật khó hơn (chất lượng truy hồi trở thành nút thắt).

### Bài kiểm tra needle-in-a-haystack

Bài kiểm tra ngữ cảnh dài chuẩn mực: chèn một dấu hiệu thị giác hoặc văn bản duy nhất vào một điểm ngẫu nhiên trong video, rồi hỏi một câu buộc phải nhớ lại dấu hiệu đó.

Chỉ số: Recall@k theo độ dài video và vị trí dấu hiệu.

Gemini 2.5 Pro đạt truy hồi >99% với video tới 90 phút. Các model mở 72B (Qwen2.5-VL-72B, InternVL3-78B) đạt ~85-90% ở 30 phút và giảm dần sau 60 phút.

VideoAgent có thể ngang hoặc hơn model ngữ cảnh thô ở video 2+ giờ, vì truy hồi chạm đúng dấu hiệu nếu công cụ đủ tốt.

### Chọn đường nào

Với clip 15 phút cần độ chính xác hàng đầu: 72B mở + ngữ cảnh gốc thường là đủ. Chọn Qwen2.5-VL-72B.

Với nội dung 30 phút đến 1 giờ: LongVILA hoặc Video-XL nếu mở; Gemini 2.5 Pro nếu đóng. Ngưỡng chất lượng quan trọng — cần hàng đầu thì đi hướng đóng.

Với nội dung 2+ giờ: VideoAgent hoặc các mẫu truy hồi tương tự. Cách khác: tóm tắt thành từng khối nhỏ hơn rồi nạp các bản tóm tắt hierarchical (phân cấp).

### Mẫu triển khai năm 2026

Trên thực tế, pipeline long video trong production là lai:

1. Chạy lấy mẫu FPS động + pooling mạnh trên toàn video (thu được biểu diễn toàn cục 100k token).
2. Đưa cho VLM 72B để tóm tắt toàn cục.
3. Nếu người dùng hỏi chi tiết, chạy agentic retrieval dùng bản tóm tắt làm chỉ mục.

Cách này kết hợp ngữ cảnh thô cho hiểu biết toàn cục và truy hồi cho chi tiết cục bộ.

```figure
mm-video-token-budget
```

## Áp dụng

`code/main.py`:

- Tính ngân sách token cho video từ 1 phút đến 3 giờ ở các mức FPS + pooling khác nhau.
- Mô phỏng một lượt needle-in-a-haystack: chèn dấu hiệu vào một mốc thời gian ngẫu nhiên, đặt câu hỏi, chấm điểm truy hồi.
- Kèm một trình mô phỏng router agentic-retrieval chọn các clip cụ thể để đưa cho VLM phía sau.

Chạy bảng ngân sách và cảm nhận khoảng cách quy mô.

## Đóng gói

Bài này tạo ra `outputs/skill-long-video-strategy-planner.md`. Cho trước độ dài video và độ phức tạp truy vấn, nó chọn giữa ngữ cảnh thô, nén và agentic retrieval, đồng thời tính độ trễ + kỳ vọng chất lượng.

## Bài tập

1. Một bài giảng 45 phút ở 1 FPS, 81 token mỗi khung. Tổng token? Vừa với ngữ cảnh của những model nào?

2. Thiết kế một bài kiểm tra needle-in-a-haystack: bạn chèn dấu hiệu ở phút thứ mấy, và định dạng truy vấn chính xác là gì?

3. So sánh Qwen2.5-VL-72B ngữ cảnh thô (ngữ cảnh 80k) với VideoAgent (Claude 3.5 + truy hồi) trên video 1 giờ. Cái nào thắng về truy hồi? Cái nào thắng về độ trễ?

4. Chi phí bộ nhớ của ring attention tăng tuyến tính theo độ dài chuỗi và tuyến tính theo số thiết bị. Giải thích vì sao, và điều gì hỏng nếu bạn bỏ pha xoay vòng.

5. Đọc Mục 5 của Gemini 1.5 về needle-in-a-haystack. Bài báo phát hiện gì về truy hồi ở ranh giới 1M so với 10M token?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Brute context | "Chỉ thêm token" | Mở ngữ cảnh LLM lên hàng triệu token; xử lý mọi thứ trong một lượt |
| Ring attention | "Song song kiểu LWM" | Mẫu attention phân tán, mỗi thiết bị giữ một đoạn và luân chuyển |
| Token compression | "Token tóm tắt" | Giảm token mỗi clip bằng một bộ nén học được trước khi đưa vào LLM |
| Needle-in-haystack | "Bài test NIH" | Chèn một dấu hiệu duy nhất ở điểm ngẫu nhiên, hỏi model nhớ lại lúc kiểm tra |
| Agentic retrieval | "LLM làm người lập truy vấn" | LLM hỏi công cụ truy hồi lấy clip liên quan, đọc chúng qua VLM, ghép câu trả lời |
| VideoAgent | "Mẫu truy hồi cho video" | Thiết kế agentic-retrieval kinh điển: câu hỏi -> công cụ -> clip -> câu trả lời |

## Đọc thêm

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)

---

[← Bài trước: Mô hình video-ngôn ngữ (video-language): token thời gian và định vị theo thời gian](../../17-video-language-temporal-grounding/docs/vi.md) · [Bài sau: Mô hình âm thanh-ngôn ngữ: cung đường từ Whisper tới Audio Flamingo 3 →](../../19-audio-language-whisper-to-af3/docs/vi.md)
