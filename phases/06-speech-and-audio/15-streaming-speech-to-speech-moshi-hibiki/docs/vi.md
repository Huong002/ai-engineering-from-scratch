# Streaming speech-to-speech — Moshi, Hibiki, và hội thoại song công toàn phần

> Giai đoạn 2024-2026 đã định nghĩa lại AI giọng nói. Moshi tung ra một mô hình duy nhất vừa nghe vừa nói cùng lúc với độ trễ 200 ms. Hibiki làm dịch speech-to-speech (giọng nói sang giọng nói) theo từng đoạn. Cả hai từ bỏ pipeline ASR → LLM → TTS để dùng kiến trúc full-duplex (song công toàn phần) thống nhất trên các codec token (token codec) của Mimi. Đây là thiết kế tham chiếu mới.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 13 (Codec âm thanh neural), Giai đoạn 6, Bài 11 (Âm thanh thời gian thực), Giai đoạn 7, Bài 05 (Transformer đầy đủ)
**Time:** ~75 phút

## Vấn đề

Mọi voice agent xây từ Bài 11 + 12 đều có sàn độ trễ cơ bản khoảng 300-500 ms: VAD kích hoạt, STT xử lý, LLM suy luận, TTS sinh. Mỗi tầng có độ trễ tối thiểu riêng. Bạn có thể tinh chỉnh và song song hoá, nhưng hình dạng pipeline sẽ chặn trần của bạn.

Moshi (Kyutai, 2024-2026) đặt một câu hỏi khác: nếu không có pipeline thì sao? Nếu một mô hình nhận âm thanh vào và phát âm thanh ra trực tiếp, liên tục, với văn bản chỉ là "độc thoại nội tâm" trung gian thay vì một tầng bắt buộc thì sao?

Câu trả lời là **full-duplex speech-to-speech**. Độ trễ lý thuyết 160 ms (80 ms khung Mimi + 80 ms delay (độ trễ) âm học). Độ trễ thực tế 200 ms trên một GPU L4. Bằng một nửa so với voice agent dạng pipeline tốt nhất.

## Khái niệm

![Kiến trúc Moshi: hai luồng Mimi song song + văn bản độc thoại nội tâm](../assets/moshi-hibiki.svg)

### Kiến trúc Moshi

**Đầu vào.** Hai luồng codec Mimi, cùng ở 12.5 Hz × 8 codebook:

- Luồng 1: âm thanh người dùng (được mã hoá bằng Mimi, đến liên tục)
- Luồng 2: âm thanh của chính Moshi (do Moshi sinh ra)

**Transformer.** Một Temporal Transformer 7B tham số xử lý cả hai luồng và một luồng văn bản "độc thoại nội tâm". Ở mỗi bước 80 ms, nó:

1. Tiêu thụ các token Mimi mới nhất của người dùng (8 codebook).
2. Tiêu thụ các token Mimi gần nhất của Moshi (8 codebook, ngay khi được sinh ra).
3. Sinh token văn bản tiếp theo của Moshi (độc thoại nội tâm).
4. Sinh các token Mimi tiếp theo của Moshi (8 codebook qua một Depth Transformer nhỏ).

Cả ba luồng — âm thanh người dùng, âm thanh Moshi, văn bản Moshi — chạy song song. Moshi có thể nghe người dùng trong khi đang nói; có thể tự ngắt khi người dùng ngắt lời; có thể back-channel ("mhm") mà không phá vỡ lượt nói (turn-taking — luân phiên lượt) chính của mình.

**Depth transformer.** Trong một khung, 8 codebook không được dự đoán song song — chúng có phụ thuộc liên-codebook. Một "depth transformer" nhỏ 2 lớp dự đoán chúng tuần tự trong 80 ms. Đây là cách phân rã chuẩn cho các AR codec LM (VALL-E, VibeVoice cũng dùng).

### Vì sao văn bản độc thoại nội tâm có ích

Không có văn bản tường minh, mô hình phải ngầm mô hình hoá ngôn ngữ ngay trong luồng âm học của nó. Ý tưởng của Moshi: buộc nó phát token văn bản song song với âm thanh. Luồng văn bản thực chất là bản ghi lời Moshi đang nói. Điều này cải thiện mạch lạc ngữ nghĩa, giúp thay đầu language model dễ hơn, và cho bạn bản ghi miễn phí.

### Hibiki: dịch speech-to-speech dạng streaming (truyền trực tiếp)

Cùng kiến trúc, huấn luyện trên các cặp dịch. Âm thanh nguồn vào, âm thanh ngôn ngữ đích ra, liên tục. Hibiki-Zero (tháng 2 năm 2026) loại bỏ nhu cầu dữ liệu huấn luyện căn chỉnh ở mức từ — dùng dữ liệu mức câu + học tăng cường GRPO để tối ưu độ trễ.

Ban đầu hỗ trợ bốn cặp ngôn ngữ; có thể thích ứng sang ngôn ngữ mới với ≈1000 giờ.

### Toàn bộ stack Kyutai (2026)

- **Moshi** — hội thoại song công toàn phần (tiếng Pháp trước, tiếng Anh được hỗ trợ tốt)
- **Hibiki / Hibiki-Zero** — dịch giọng nói đồng thời
- **Kyutai STT** — ASR truyền trực tiếp (nhìn trước 500 ms hoặc 2.5 s)
- **Kyutai Pocket TTS** — TTS 100M tham số chạy trên CPU (tháng 1 năm 2026)
- **Unmute** — pipeline đầy đủ kết hợp những thứ này trên các server công khai

Thông lượng trên một GPU L40S: 64 phiên đồng thời ở tốc độ gấp 3 lần thời gian thực.

### Sesame CSM — người bà con

Sesame CSM (2025) dùng ý tưởng tương tự — xương sống Llama-3 với đầu codec Mimi. Nhưng CSM là một chiều (nhận ngữ cảnh + văn bản, sinh giọng nói) chứ không phải song công toàn phần. Đây là TTS "hiện diện giọng nói" tốt nhất trên thị trường; không hẳn giống khả năng song công toàn phần của Moshi.

### Số liệu hiệu năng 2026

| Mô hình | Độ trễ | Trường hợp dùng | Giấy phép |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | hội thoại song công toàn phần tiếng Anh / tiếng Pháp | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | dịch truyền trực tiếp Pháp ↔ Anh | CC-BY 4.0 |
| Hibiki-Zero | như trên | 5 cặp ngôn ngữ, không cần dữ liệu căn chỉnh | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | TTS điều kiện hoá theo ngữ cảnh | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | đóng, OpenAI API | thương mại |
| Gemini 2.5 Live | ~350 ms | đóng, Google API | thương mại |

```figure
sp-fullduplex
```

## Tự tay làm

### Bước 1: giao diện

Moshi mở một server WebSocket nhận các đoạn âm thanh mã hoá Mimi dài 80 ms và trả về các đoạn âm thanh mã hoá Mimi dài 80 ms. Cả hai chiều. Liên tục.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Bước 2: vòng lặp song công toàn phần

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

Cả hai chiều chạy đồng thời. Python asyncio hoặc Rust futures là transport chuẩn.

### Bước 3: mục tiêu huấn luyện (khái niệm)

Với mỗi khung 80 ms `t`:

- Đầu vào: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- Dự đoán: `moshi_text[t]`, sau đó `moshi_mimi[t, codebook_0..7]`

Văn bản được dự đoán trước âm thanh (độc thoại nội tâm); âm thanh được dự đoán tuần tự theo codebook trong depth transformer.

### Bước 4: Moshi thắng ở đâu và không thắng ở đâu

Moshi thắng:

- Dưới 250 ms đầu-cuối trên phần cứng rẻ.
- Back-channel và ngắt lời tự nhiên.
- Không cần code keo nối pipeline.

Moshi không thắng:

- Gọi tool (không được huấn luyện cho việc này; bạn cần một đường LLM riêng).
- Suy luận dài (Moshi là mô hình hội thoại cỡ 8B, không phải Claude/GPT-4).
- Độ chính xác sự thật về chủ đề ngách.
- Hầu hết use case doanh nghiệp trong sản xuất (năm 2026 vẫn dùng pipeline).

## Áp dụng

| Tình huống | Lựa chọn |
|-----------|------|
| Bạn đồng hành giọng nói độ trễ thấp nhất | Moshi |
| Cuộc gọi dịch trực tiếp | Hibiki |
| Demo giọng nói / nghiên cứu | Moshi, CSM |
| Agent doanh nghiệp có tool | Pipeline (Bài 12), không phải Moshi |
| TTS giọng tuỳ chỉnh theo ngữ cảnh | Sesame CSM |
| Speech-to-speech, mọi ngôn ngữ | GPT-4o Realtime hoặc Gemini 2.5 Live (thương mại) |

## Cạm bẫy

- **Gọi tool hạn chế.** Moshi là mô hình hội thoại, không phải framework agent. Kết hợp với pipeline khi cần tool.
- **Điều kiện hoá giọng cụ thể.** Moshi dùng một persona đã huấn luyện duy nhất; nhân bản giọng là một lần huấn luyện riêng.
- **Độ phủ ngôn ngữ.** Pháp + Anh rất tốt; các ngôn ngữ khác hạn chế. Hibiki-Zero giúp được, nhưng bạn vẫn cần dữ liệu huấn luyện.
- **Chi phí tài nguyên.** Một phiên Moshi đầy đủ chiếm một slot GPU; không phải kiểu triển khai chia sẻ tenant giá rẻ.

## Đóng gói

Lưu thành `outputs/skill-duplex-pipeline.md`. Chọn pipeline hay kiến trúc song công toàn phần cho một workload voice agent, kèm lý do.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó mô phỏng kiến trúc hai luồng + độc thoại nội tâm một cách tượng trưng.
2. **Trung bình.** Kéo Moshi từ HuggingFace, chạy server, thử một cuộc hội thoại. Đo độ trễ thời gian thực từ lúc người dùng nói xong đến lúc Moshi bắt đầu trả lời.
3. **Khó.** Lấy agent pipeline ở Bài 12 của bạn và so sánh độ trễ P50 với Moshi trên 20 câu kiểm thử tương ứng. Viết ra khi nào pipeline vẫn thắng về mặt kiến trúc.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Full-duplex | Nghe và nói cùng lúc | Hai luồng âm thanh hoạt động đồng thời trên cùng một mô hình. |
| Inner monologue | Luồng văn bản của mô hình | Moshi phát token văn bản song song với đầu ra âm thanh. |
| Depth transformer | Bộ dự đoán liên-codebook | Transformer nhỏ dự đoán 8 codebook trong một khung 80 ms. |
| Mimi | Codec của Kyutai | 12.5 Hz × 8 codebook; semantic token (token ngữ nghĩa)+âm học; là động cơ của Moshi. |
| Streaming S2S | Âm thanh → âm thanh trực tiếp | Dịch/hội thoại từng đoạn, không có tầng pipeline. |
| Back-channeling | Phản ứng "Mhm" | Moshi có thể phát những lời xác nhận nhỏ mà không cắt lượt của mình. |

## Đọc thêm

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — bài báo.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — dịch truyền trực tiếp không cần dữ liệu căn chỉnh.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — đặc tả CSM.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — cài đặt + server.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — đối thủ thương mại đóng.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — framework STT/TTS bên dưới.

---

[← Bài trước: Voice Activity Detection (phát hiện hoạt động tiếng nói) & Turn-Taking (luân phiên lượt nói)](../../14-voice-activity-detection-turn-taking/docs/vi.md) · [Bài sau: Chống giả mạo giọng nói & dấu thuỷ văn âm thanh →](../../16-anti-spoofing-audio-watermarking/docs/vi.md)
