# Mô hình âm thanh-ngôn ngữ: cung đường từ Whisper tới Audio Flamingo 3

> Whisper (Radford et al., tháng 12 năm 2022) đã khép lại bài toán nhận dạng giọng nói (speech recognition) — 680k giờ giọng nói đa ngôn ngữ được giám sát yếu, một transformer encoder-decoder đơn giản, một bộ đánh giá khiến mọi bản phát hành ASR sau đó đều phải dẫn lại. Nhưng nhận dạng không phải suy luận. Hỏi "bản thu này có những nhạc cụ gì" hay "người nói đang thể hiện cảm xúc gì" hay "phút thứ 3 đã xảy ra chuyện gì" đòi hỏi hiểu âm thanh chứ không chỉ là bản ghi (transcription). Qwen-Audio, SALMONN, LTU và Audio Flamingo 3 của NVIDIA (AF3, tháng 7 năm 2025) đã lần lượt dựng nên chồng công nghệ ấy: giữ bộ mã hoá âm thanh (audio encoder) kiểu Whisper, gắn thêm Q-former, huấn luyện trên dữ liệu chỉ dẫn âm thanh-văn bản, thêm suy luận chain-of-thought. Bài học này đi trọn cung đường đó.

**Type:** Build
**Languages:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites:** Giai đoạn 6 (Giọng nói và âm thanh), Giai đoạn 12, Bài 03 (Q-Former)
**Time:** ~180 phút

## Mục tiêu học tập

- Tính log-Mel spectrogram (phổ mel) từ waveform: chia cửa sổ, FFT, dải lọc, biến đổi log.
- So sánh các lựa chọn encoder: Whisper encoder, BEATs, bản lai AF-Whisper. Khi nào mỗi loại thắng.
- Dựng một audio Q-former: N query học được cross-attend lên các patch của spectrogram.
- Giải thích huấn luyện ghép tầng (Whisper-rồi-LLM) so với LLM âm thanh (audio LLM) đầu-cuối: vì sao đầu-cuối mở rộng tốt hơn cho suy luận.

## Vấn đề

Nhận dạng giọng nói đã được Whisper giải quyết. OCR-của-âm-thanh là hàng phổ thông. Nhưng "phổ thông" dừng lại ở bản ghi. Nếu mô hình không suy luận được trên thứ nó vừa nghe — thời điểm, người nói, cảm xúc, cấu trúc âm nhạc, âm thanh môi trường — thì chỉ bản ghi không đủ để tạo ra tính năng sản phẩm.

Ba con đường hiển nhiên:

1. Ghép tầng: Whisper tạo bản ghi, LLM suy luận trên bản ghi ấy. Hợp với kịch bản thuần lời nói. Thất bại với nhạc, âm thanh môi trường, nhiều người nói chồng tiếng, cảm xúc.

2. LLM âm thanh đầu-cuối: một audio encoder đưa token âm thanh thẳng vào LLM, bỏ qua bước tạo bản ghi. Giữ được thông tin âm học (cảm xúc, người nói, môi trường). Cần dữ liệu huấn luyện mới.

3. Lai: audio encoder + text decoder vừa tạo bản ghi vừa suy luận. Qwen-Audio và Audio Flamingo chọn con đường này.

## Khái niệm

### Log-Mel spectrogram: đặc trưng đầu vào

Mọi audio encoder đều bắt đầu từ cùng một đặc trưng: một log-Mel spectrogram.

1. Lấy mẫu lại về 16 kHz — tần số lấy mẫu (sampling rate) chuẩn.
2. Biến đổi Fourier thời gian ngắn với cửa sổ 25ms, hop 10ms.
3. Lấy độ lớn của kết quả FFT.
4. Áp các dải lọc Mel (thường 80 bộ lọc dàn theo thang log 0-8000 Hz) để bẻ cong sang thang tần số cảm nhận.
5. Nén log (log(1 + x)) cho dải động.

Kết quả: một mảng 2 chiều kích thước (T, 80), trong đó T là số khung thời gian (frame). Với clip 30 giây ở tốc độ 100 khung/giây: (3000, 80).

### Encoder của Whisper

Encoder của Whisper là một transformer kiểu ViT 12 lớp, xử lý log-Mel spectrogram như một chuỗi khung thời gian. Đầu ra: một vector hidden-state cho mỗi khung thời gian.

Với ASR, decoder của Whisper là transformer cross-attention, sinh token văn bản có điều kiện trên đầu ra của encoder. Kiến trúc encoder-decoder chuẩn.

Với ALM (audio-LLM), bạn muốn đầu ra encoder làm đầu vào cho một LLM khác. Công thức: encoder Whisper đóng băng, Q-former huấn luyện được, LLM đóng băng hoặc tinh chỉnh.

### BEATs và các encoder chuyên cho âm thanh

Whisper được huấn luyện trên dữ liệu thiên về lời nói. Nó yếu hơn với nhạc và âm thanh môi trường.

BEATs (Chen et al., 2022) là transformer tự giám sát huấn luyện trên AudioSet. Nắm nhạc và âm thanh môi trường tốt hơn Whisper ở cùng số tham số.

AF-Whisper (bản lai của Audio Flamingo 3): nối đặc trưng Whisper + BEATs làm đầu vào âm thanh. Whisper mang tín hiệu ngôn ngữ, BEATs mang tín hiệu âm học.

### Audio Q-former

Cùng khuôn mẫu với visual Q-former của BLIP-2. Một số cố định query học được (thường 32 hoặc 64) cross-attend lên các khung đầu ra của audio encoder. Các query trở thành token âm thanh mà LLM tiêu thụ.

Giai đoạn căn chỉnh khi huấn luyện: chỉ Q-former, loss contrastive + captioning trên các cặp âm thanh-văn bản (AudioCaps, Clotho). Giai đoạn chỉ dẫn: đầu-cuối, mở băng LLM, huấn luyện trên dữ liệu chỉ dẫn.

### Cung đường — SALMONN, Qwen-Audio, AF3

SALMONN (Tang et al., 2023): Whisper + BEATs + Q-former + LLaMA. LLM âm thanh mở đầu tiên có khả năng suy luận nghiêm túc. Điểm trên MMAU khoảng 0.55 tổng hợp.

Qwen-Audio (Chu et al., 2023): kiến trúc tương tự, huấn luyện trên bộ dữ liệu phong phú hơn, tinh chỉnh cho hội thoại nhiều lượt. MMAU ~0.60.

LTU — Listen, Think, Understand (Gong et al., 2023): dữ liệu suy luận tường minh, tập trung vào chain-of-thought trên các clip âm thanh. Nhỏ hơn nhưng tập trung hơn.

Audio Flamingo 3 (Goel et al., tháng 7 năm 2025): SOTA mở hiện tại. Xương sống LLM 8B (Qwen2 7B), encoder Whisper-large nối BEATs, Q-former 64 query, huấn luyện trên hơn 1M cặp chỉ dẫn âm thanh-văn bản. MMAU 0.72, ngang các mô hình độc quyền tiên phong ở một số bài con.

AF3 còn giới thiệu chain-of-thought theo yêu cầu cho âm thanh: mô hình có thể tuỳ chọn phát ra token suy nghĩ ("để tôi xác định nhạc cụ trước: ...") trước câu trả lời cuối. Độ chính xác trên các bài suy luận phức tạp tăng 3-5 điểm khi bật suy nghĩ.

### Ghép tầng so với đầu-cuối

Pipeline ghép tầng:

1. Whisper tạo bản ghi âm thanh → văn bản.
2. LLM suy luận trên văn bản.

Chạy hoàn hảo với "tóm tắt podcast này." Thất bại với:
- "Tâm trạng của bài hát này là gì?" — tâm trạng nằm trong âm thanh, không phải lời.
- "Ai đang nói, Alice hay Bob?" — cần nhận diện người nói.
- "Vụ nổ xảy ra ở giây thứ mấy?" — mốc thời gian bị mất trong văn bản.
- "Đây là âm thanh thật hay do máy sinh?" — phát hiện deepfake cần đặc trưng âm học.

Đầu-cuối giữ được tín hiệu âm học. Qwen-Audio và AF3 xử lý nhạc, môi trường và cảm xúc một cách tự nhiên.

### Công thức triển khai 2026

Với một sản phẩm hiểu âm thanh mới:

- Ghép tầng nếu: mục tiêu là bản ghi, không có nhạc, không suy luận cảm xúc.
- AF3 / họ Qwen-Audio nếu: có nhạc, cảm xúc, nhiều người nói, hoặc suy luận âm thanh phức tạp.

Ghép tầng rẻ hơn và đơn giản hơn. Đầu-cuối mạnh hơn.

### MMAU — bộ đánh giá suy luận âm thanh

MMAU (Massive Multimodal Audio Understanding) là bộ đánh giá suy luận âm thanh giai đoạn 2024-2025:

- 10,000 cặp QA âm thanh-văn bản trải trên lời nói, nhạc, âm thanh môi trường.
- Bao phủ phân loại, suy luận thời gian, suy luận nhân quả, QA mở.
- Kiểm tra đúng những thứ pipeline ghép tầng bỏ sót một cách hệ thống.

SOTA mở (AF3) ở 0.72; biên độc quyền ~0.78 (Gemini 2.5 Pro, Claude Opus 4.7). Khoảng cách nhỏ hơn delta mở-vs-đóng của VideoMME, cho thấy LLM âm thanh đang chín dần.

```figure
audio-text-ctc
```

## Áp dụng

`code/main.py`:

- Cài đặt tính log-Mel spectrogram bằng stdlib: chia cửa sổ, DFT ngây thơ, dải lọc Mel.
- Bộ khung audio Q-former: từ các khung đầu ra encoder, tính Q, K, V, attention và phát ra N token.
- So sánh ghép tầng-vs-đầu-cuối trên một bài toán đồ chơi.

## Đóng gói

Bài học này tạo ra `outputs/skill-audio-llm-pipeline-picker.md`. Cho một bài toán âm thanh (bản ghi, gắn nhãn nhạc, suy luận cảm xúc, phân tách người nói, phân loại môi trường), nó chọn ghép tầng, AF3 đầu-cuối, hay bản lai.

## Bài tập

1. Tính kích thước log-Mel spectrogram cho clip 30 giây ở 16kHz, cửa sổ 25ms, hop 10ms, 80 bin Mel. Điều này thay đổi thế nào ở 48kHz?

2. Vì sao Whisper kém hơn trên nhạc? BEATs nắm được đặc trưng âm thanh nào mà Whisper không?

3. Audio Q-former với 64 query so với 32: ở độ phức tạp bài toán nào thì 64 đáng giá? 32 tiết kiệm tính toán cho việc gì?

4. Đọc Mục 4 của AF3 về suy nghĩ theo yêu cầu. Đề xuất ba bài toán âm thanh mà chain-of-thought giúp nhiều nhất.

5. Cài đặt một pipeline phân tách người nói tối giản dùng đầu ra của AF3. Bạn báo hiệu chuyển người nói thế nào?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Log-Mel spectrogram | "đặc trưng Mel" | Mảng 2 chiều (thời gian, tần số) các giá trị log-magnitude sau khi qua dải lọc Mel |
| Audio Q-former | "Audio Perceiver" | Nút cổ chai cross-attention từ đầu ra audio encoder sang các query độ dài cố định đưa vào LLM |
| Cascaded | "ASR-rồi-LLM" | Pipeline trong đó Whisper tạo bản ghi và một LLM văn bản suy luận; làm mất thông tin âm học |
| End-to-end | "Audio-LLM" | Đặc trưng âm thanh vào thẳng LLM qua Q-former; giữ được tín hiệu âm học |
| BEATs | "encoder AudioSet cho âm thanh" | Transformer SSL huấn luyện trên AudioSet; mạnh ở nhạc + âm thanh môi trường |
| MMAU | "bộ đánh giá suy luận âm thanh" | 10k cặp QA trên lời nói, nhạc, môi trường; chuẩn đánh giá 2024 |
| On-demand thinking | "CoT cho âm thanh" | Mô hình có thể tuỳ chọn phát token suy luận trước câu trả lời cuối, tăng độ chính xác 3-5 điểm |

## Đọc thêm

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)

---

[← Bài trước: Hiểu long video (video dài) ở cửa sổ ngữ cảnh hàng triệu token](../../18-long-video-million-token/docs/vi.md) · [Bài sau: Omni Models: Qwen2.5-Omni và cách tách Thinker-Talker →](../../20-omni-models-thinker-talker/docs/vi.md)
