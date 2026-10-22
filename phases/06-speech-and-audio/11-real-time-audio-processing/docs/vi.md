# Xử lý âm thanh trong thời gian thực

> Pipeline xử lý theo lô thì xử lý một tệp. Pipeline real-time (thời gian thực) thì xử lý 20 mili giây tiếp theo trước khi 20 mili giây kế tiếp ập đến. Mọi AI hội thoại, studio phát sóng và bot tổng đài đều sống chết theo ngân sách độ trễ (latency) này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 02 (Spectrogram), Giai đoạn 6, Bài 04 (ASR), Giai đoạn 6, Bài 07 (TTS)
**Time:** ~75 phút

## Vấn đề

Bạn muốn một trợ lý giọng nói có cảm giác sống động. Độ trễ chuyển lượt trong hội thoại của người là ~230 ms (từ lúc im lặng đến lúc phản hồi). Trên 500 ms thì nghe máy móc; trên 1500 ms thì nghe như đã hỏng. Ngân sách cho một vòng **nghe → hiểu → phản hồi → nói** trọn vẹn trong năm 2026 là:

| Công đoạn | Ngân sách |
|-------|--------|
| Mic → bộ đệm (buffer) | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (token đầu tiên) | 100 ms |
| TTS (khối đầu tiên) | 100 ms |
| Render → loa | 20 ms |
| **Tổng** | **~400 ms** |

Moshi (Kyutai, 2024) đạt 200 ms full-duplex. GPT-4o-realtime (2024) đạt ~320 ms. Các pipeline nối tầng năm 2022 chạy ở 2500 ms. Cải thiện 10× đến từ ba kỹ thuật: (1) streaming ở mọi tầng, (2) pipeline bất đồng bộ với kết quả một phần, (3) sinh có thể ngắt.

## Khái niệm

![Pipeline âm thanh streaming với bộ đệm vòng, cổng VAD, ngắt lời](../assets/real-time.svg)

**Frame / chunk (khối dữ liệu) / window.** Âm thanh thời gian thực chảy thành các khối kích thước cố định. Lựa chọn phổ biến: 20 ms (320 mẫu ở 16 kHz). Mọi thứ phía sau phải theo kịp nhịp này.

**Ring buffer (bộ đệm vòng).** Bộ đệm tròn kích thước cố định. Luồng producer ghi frame mới, luồng consumer đọc. Tránh cấp phát bộ nhớ trên đường nóng. Kích thước ≈ độ trễ tối đa × tần số lấy mẫu; một vòng 2 giây ở 16 kHz = 32.000 mẫu.

**VAD (Voice Activity Detection).** Đóng mở công việc phía sau khi không ai nói. Silero VAD 4.0 (2024) chạy <1 ms mỗi frame 30 ms trên CPU. `webrtcvad` là lựa chọn cũ hơn.

**Streaming ASR.** Các model phát ra bản ghi một phần ngay khi âm thanh đến. Parakeet-CTC-0.6B ở chế độ streaming (NeMo, 2024) đạt WER 2–5% với độ trễ 320 ms. Whisper-Streaming (Macháček et al., 2023) chia Whisper thành khối để gần-streaming với độ trễ ~2 s.

**Interruption.** Khi người dùng nói trong lúc trợ lý đang nói, bạn phải (a) phát hiện cú ngắt lời (barge-in), (b) dừng TTS, (c) bỏ phần còn lại của đầu ra LLM. Tất cả trong 100 ms, nếu không người dùng sẽ thấy trợ lý như bị điếc.

**Transport WebRTC Opus.** Frame 20 ms, 48 kHz, bitrate thích ứng 8–128 kbps. Chuẩn cho trình duyệt và di động. LiveKit, Daily.co, Pion là các bộ stack 2026 để làm ứng dụng giọng nói.

**Jitter buffer.** Gói mạng đến sai thứ tự hoặc đến muộn. Bộ đệm jitter (jitter — dao động thời điểm gói đến) sắp xếp lại và làm mượt; quá nhỏ → nghe thấy khoảng hở, quá lớn → tăng độ trễ. Thường 60–80 ms.

### Những cái bẫy phổ biến

- **Tranh chấp luồng.** GIL của Python + model nặng có thể bỏ đói luồng âm thanh. Dùng thư viện âm thanh có callback (hàm gọi lại) ở tầng C (sounddevice, PortAudio) và giữ Python ra khỏi đường nóng.
- **Độ trễ chuyển đổi tần số lấy mẫu.** Resample ngay trong pipeline cộng thêm 5–20 ms. Hoặc resample từ đầu, hoặc dùng resampler không độ trễ (PolyPhase, `soxr_hq`).
- **Khởi động TTS.** Cả TTS nhanh như Kokoro cũng mất 100–200 ms khởi động ở lần gọi đầu. Cache model + chạy thử một lần rỗng trước lượt thật đầu tiên.
- **Khử vọng.** Không có AEC, đầu ra TTS lại vào mic và kích hoạt ASR trên chính giọng của bot. WebRTC AEC3 là lựa chọn mặc định mã nguồn mở.

```figure
nyquist-aliasing
```

## Tự tay làm

### Bước 1: bộ đệm vòng

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Capacity quyết định độ trễ đệm tối đa. 32.000 mẫu ở 16 kHz = 2 s.

### Bước 2: cổng VAD

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Thay bằng Silero VAD khi lên production:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Bước 3: ASR streaming

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Bước 4: bộ xử lý ngắt lời

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Điều này phụ thuộc vào async I/O và TTS streaming có thể huỷ. Gọi WebRTC peerconnection.stop() trên audio track là cách chuẩn.

## Áp dụng

Bộ stack năm 2026:

| Tầng | Lựa chọn |
|-------|------|
| Transport | LiveKit (WebRTC) hoặc Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B hoặc Whisper-Streaming |
| Token đầu tiên của LLM | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro hoặc ElevenLabs Turbo v2.5 |
| Khử vọng | WebRTC AEC3 |
| Native đầu-cuối | OpenAI Realtime API hoặc Moshi |

## Cạm bẫy

- **Đệm 500 ms cho chắc.** Bộ đệm *chính là* mức sàn độ trễ của bạn. Hãy thu nhỏ nó.
- **Không ghim luồng.** Callback âm thanh chạy trên luồng có độ ưu tiên thấp hơn UI = giật, rè khi tải cao.
- **Khối TTS quá nhỏ.** Khối dưới 200 ms làm lộ artifact của vocoder. Khối 320 ms là điểm ngọt.
- **Không có bộ đệm jitter.** Mạng thật luôn dao động; không làm mượt thì bạn sẽ nghe tiếng "pop".
- **Xử lý lỗi không có dự phòng.** Pipeline âm thanh phải chống crash. Một exception là chết cả phiên.

## Đóng gói

Lưu thành `outputs/skill-realtime-designer.md`. Thiết kế một pipeline âm thanh thời gian thực với ngân sách độ trễ cụ thể cho từng công đoạn.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Mô phỏng bộ đệm vòng + VAD theo năng lượng; in độ trễ từng công đoạn cho một luồng giả 10 giây.
2. **Trung bình.** Dùng `sounddevice`, dựng vòng lặp passthrough xử lý mic của bạn theo frame 20 ms và in trạng thái VAD ở mỗi frame.
3. **Khó.** Dựng bài test vọng full duplex với `aiortc`: browser → WebRTC → Python → WebRTC → browser. Đo độ trễ glass-to-glass bằng một xung 1 kHz.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Ring buffer | Hàng đợi vòng | FIFO kích thước cố định, lock-free (hoặc khoá SPSC) cho các frame âm thanh. |
| VAD | Cổng im lặng | Model hoặc heuristic đánh dấu đâu là tiếng nói, đâu không phải. |
| Streaming ASR | STT thời gian thực | Phát ra văn bản một phần ngay khi âm thanh đến; lookahead có giới hạn. |
| Jitter buffer | Bộ làm mượt mạng | Hàng đợi sắp xếp lại các gói sai thứ tự; thường 60–80 ms. |
| AEC | Khử vọng | Trừ đi đường phản hồi từ loa vào mic. |
| Barge-in | Người dùng ngắt lời | Hệ thống phát hiện người dùng nói khi TTS đang chạy; phải huỷ phát. |
| Full duplex | Hai chiều cùng lúc | Người dùng và bot có thể nói cùng lúc; Moshi là full duplex. |

## Đọc thêm

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — Whisper chia khối, gần-streaming.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — full-duplex, độ trễ 200 ms.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — điều phối audio agent trong production.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — VAD dưới 1 ms, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — khử vọng dưới dạng mã nguồn mở.

---

[← Bài trước: Mô hình ngôn ngữ âm thanh](../../10-audio-language-models/docs/vi.md) · [Bài sau: Xây dựng quy trình xử lý trợ lý giọng nói →](../../12-voice-assistant-pipeline/docs/vi.md)
