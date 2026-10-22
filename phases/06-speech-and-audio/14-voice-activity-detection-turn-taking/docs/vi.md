# Voice Activity Detection (phát hiện hoạt động tiếng nói) & Turn-Taking (luân phiên lượt nói) — Silero, Cobra và thủ thuật flush

> Mỗi voice agent sống hay chết nhờ hai quyết định: người dùng có đang nói không, và họ đã nói xong chưa? VAD trả lời câu thứ nhất. Turn detection (VAD + silence hangover + model semantic endpoint) trả lời câu thứ hai. Sai một trong hai là trợ lý hoặc cắt lời người dùng, hoặc không bao giờ chịu im.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 11 (Âm thanh thời gian thực), Giai đoạn 6, Bài 12 (Trợ lý giọng nói)
**Time:** ~45 phút

## Vấn đề

Ba quyết định riêng biệt mà một voice agent đưa ra trên mỗi chunk 20 ms:

1. **Frame này có phải tiếng nói không?** — VAD. Nhị phân, theo từng frame.
2. **Người dùng đã bắt đầu một utterance mới chưa?** — phát hiện onset.
3. **Người dùng đã nói xong chưa?** — endpointing (xác định điểm kết thúc), tức kết thúc lượt.

Câu trả lời ngây thơ (ngưỡng năng lượng — threshold) hỏng trước mọi loại nhiễu — xe cộ, bàn phím, tiếng đám đông. Câu trả lời của năm 2026: Silero VAD (mở, deep-learned) + một model turn detection (semantic endpointing) + silence hangover (thời gian giữ) được hiệu chỉnh theo VAD.

## Khái niệm

![Thác VAD: năng lượng → Silero → turn-detector → thủ thuật flush](../assets/vad-turn-taking.svg)

### Ba tầng của thác VAD

**Tầng 1: cổng năng lượng.** Rẻ nhất. Ngưỡng RMS ở -40 dBFS. Lọc những khoảng silence (im lặng) rõ ràng nhưng vẫn kích hoạt với bất kỳ nhiễu nào vượt ngưỡng.

**Tầng 2: Silero VAD** (2020-2026, MIT). 1M tham số. Huấn luyện trên hơn 6000 ngôn ngữ. Chạy ~1 ms cho mỗi chunk 30 ms trên một thread CPU. 87.7% TPR tại 5% FPR. Lựa chọn mặc định của mã nguồn mở.

**Tầng 3: semantic turn detector.** Model turn detection của LiveKit (2024-2026) hoặc một classifier nhỏ của riêng bạn. Phân biệt "ngừng giữa câu" với "nói xong". Dùng ngữ cảnh ngôn ngữ (ngữ điệu + các từ vừa nói), không chỉ im lặng.

### Các tham số chính và giá trị mặc định

- **Threshold (ngưỡng).** Silero xuất ra xác suất; coi là tiếng nói khi &gt; 0.5 (mặc định) hoặc &gt; 0.3 (nhạy). Ngưỡng thấp hơn = ít cắt chữ đầu hơn, nhưng nhiều false trigger (kích hoạt nhầm) hơn.
- **Thời lượng tiếng nói tối thiểu.** Loại bỏ tiếng nói ngắn hơn 250 ms — thường là tiếng ho hay tiếng ghế.
- **Silence hangover (endpointing).** Sau khi VAD trả về 0, chờ 500-800 ms rồi mới tuyên bố hết lượt. Quá ngắn → cắt lời người dùng. Quá dài → cảm giác ì.
- **Pre-roll buffer.** Giữ 300-500 ms âm thanh trước khi VAD kích hoạt. Tránh việc "hey" bị cắt mất.

### Thủ thuật flush (Kyutai 2025)

Các model STT streaming có độ trễ look-ahead (500 ms với Kyutai STT-1B, 2.5 s với STT-2.6B). Bình thường bạn phải chờ đúng khoảng đó sau khi hết tiếng nói mới có transcript. Thủ thuật flush: khi VAD báo hết tiếng nói, **gửi tín hiệu flush cho STT** để buộc nó xuất kết quả ngay. STT xử lý nhanh gấp ~4× thời gian thực, nên buffer 500 ms xong trong ~125 ms.

End-to-end: 125 ms VAD + flush STT = độ trễ hội thoại.

### So sánh VAD năm 2026

| VAD | TPR @ 5% FPR | Latency | License |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero là mặc định hợp lý. Cobra là bản nâng cấp về compliance / độ chính xác. VAD chỉ dựa trên năng lượng không còn chỗ trong production năm 2026.

```figure
sp-vad-cascade
```

## Tự tay làm

### Bước 1: cổng năng lượng

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Bước 2: Silero VAD trong Python

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Bước 3: máy trạng thái kết thúc lượt

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Bước 4: khung xương của thủ thuật flush

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT (Kyutai, Deepgram, AssemblyAI) phải hỗ trợ flush thì cách này mới chạy. Whisper streaming thì không — nó chạy theo block và luôn chờ đủ chunk.

## Áp dụng

| Tình huống | Lựa chọn VAD |
|-----------|-----------|
| Mở, nhanh, tổng quát | Silero VAD |
| Call center thương mại | Cobra VAD |
| Trên thiết bị (điện thoại) | Silero VAD ONNX |
| Nghiên cứu / diarization | pyannote segmentation |
| Phương án dự phòng không phụ thuộc | WebRTC VAD (legacy) |
| Cần chất lượng kết thúc lượt | Silero + turn-detector của LiveKit, xếp lớp |

Nguyên tắc: đừng bao giờ ship VAD chỉ dựa trên năng lượng trừ khi bạn thực sự không còn lựa chọn nào khác.

## Cạm bẫy

- **Ngưỡng cố định.** Chạy tốt khi yên tĩnh, hỏng khi ồn. Hoặc hiệu chỉnh trên thiết bị, hoặc chuyển sang Silero.
- **Hangover quá ngắn.** Agent cắt lời giữa câu. 500-800 ms là điểm ngọt cho lời nói hội thoại.
- **Hangover quá dài.** Cảm giác ì. Hãy A/B test với người dùng mục tiêu.
- **Không có pre-roll buffer.** Mất 200-300 ms âm thanh đầu tiên của người dùng. Luôn giữ một pre-roll cuốn theo.
- **Bỏ qua semantic endpointing.** "Hmm, để tôi nghĩ đã..." có những khoảng ngừng dài. Người dùng ghét bị cắt ngang giữa dòng suy nghĩ. Dùng turn-detector của LiveKit hoặc tương tự.

## Đóng gói

Lưu thành `outputs/skill-vad-tuner.md`. Chọn model VAD, ngưỡng, hangover, pre-roll và chiến lược turn detection cho một workload.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Script mô phỏng chuỗi nói + im lặng + nói + tiếng ho và kiểm tra ba tầng VAD.
2. **Trung bình.** Cài `silero-vad`, xử lý một bản ghi 5 phút, chỉnh ngưỡng để giảm cả việc cắt chữ đầu lẫn kích hoạt nhầm. Báo cáo precision/recall.
3. **Khó.** Xây một mini turn-detector: Silero VAD + MLP 3 lớp trên embedding của 10 từ cuối (dùng sentence-transformers). Huấn luyện trên một dataset kết thúc lượt được gán nhãn thủ công. Vượt Silero-only 10% F1.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| VAD | Voice detector | Nhị phân theo từng frame: đây có phải tiếng nói? |
| Turn detection | Endpointing | VAD + silence hangover + semantic endpoint. |
| Silence hangover | Wait-after-speech | Thời gian chờ trước khi tuyên bố hết lượt; 500-800 ms. |
| Pre-roll | Pre-speech buffer | Giữ 300-500 ms âm thanh trước khi VAD kích hoạt. |
| Flush trick | Kyutai hack | VAD → flush-STT → 125 ms thay vì độ trễ 500 ms. |
| Semantic endpoint | "Họ có ý định dừng không?" | Classifier ML xét từ ngữ, không chỉ im lặng. |
| TPR @ FPR 5% | ROC point | Benchmark VAD chuẩn; 87.7% với Silero, 50% với WebRTC. |

## Đọc thêm

- [Silero VAD](https://github.com/snakers4/silero-vad) — VAD mở làm tham chiếu.
- [Picovoice Cobra VAD](https://picovoice.ai/products/voice/voice-activity-detection/) — đầu bảng về độ chính xác thương mại.
- [Kyutai — Unmute + thủ thuật flush](https://kyutai.org/stt) — mẹo kỹ thuật dưới 200 ms.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — semantic endpointing trong production.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — baseline legacy.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — segmentation chuẩn cho diarization.

---

[← Bài trước: Neural Audio Codecs](../../13-neural-audio-codecs/docs/vi.md) · [Bài sau: Streaming speech-to-speech →](../../15-streaming-speech-to-speech-moshi-hibiki/docs/vi.md)
