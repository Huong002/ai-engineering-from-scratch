# Xây dựng quy trình xử lý trợ lý giọng nói — Bài tổng kết Giai đoạn 6

> Mọi thứ từ bài 01-11, được khâu lại với nhau. Xây dựng một voice assistant (trợ lý giọng nói) biết lắng nghe, suy luận và đáp lời. Năm 2026, đây là bài toán kỹ thuật đã có lời giải, không còn là bài toán nghiên cứu — nhưng chi tiết tích hợp mới quyết định nó có ra sản phẩm được hay không.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 6, Bài 04, 05, 06, 07, 11; Giai đoạn 11, Bài 09 (Function calling); Giai đoạn 14, Bài 01 (Vòng lặp agent)
**Time:** ~120 phút

## Vấn đề

Xây dựng một trợ lý end-to-end (đầu-cuối):

1. Thu âm đầu vào từ mic (16 kHz mono).
2. Phát hiện điểm bắt đầu/kết thúc giọng nói của người dùng.
3. Phiên âm theo luồng.
4. Chuyển bản phiên âm cho một LLM có khả năng gọi công cụ (hẹn giờ, thời tiết, lịch).
5. Truyền văn bản của LLM theo luồng sang TTS.
6. Phát âm thanh trở lại cho người dùng.
7. Dừng lại nếu người dùng ngắt lời giữa lúc đang trả lời.

Mục tiêu độ trễ: byte âm thanh TTS đầu tiên trong vòng 800 ms kể từ khi người dùng nói xong, trên CPU của một laptop. Mục tiêu chất lượng: không mất từ, không bịa phụ đề khi im lặng, không rò rỉ giọng nói bị sao chép, không bị prompt injection thành công.

## Khái niệm

![Quy trình xử lý trợ lý giọng nói: mic → VAD → STT → LLM+công cụ → TTS → loa](../assets/voice-assistant.svg)

### Bảy thành phần

1. **Thu âm.** Mic → 16 kHz mono → khối 20 ms. Thường dùng `sounddevice` trong Python hoặc AudioUnit/ALSA/WASAPI gốc khi lên production.
2. **VAD (Bài 11).** Silero VAD @ ngưỡng 0.5, giọng nói tối thiểu 250 ms, treo im lặng 500 ms. Phát tín hiệu "bắt đầu" và "kết thúc".
3. **STT theo luồng (Bài 4-5).** Whisper-streaming, Parakeet-TDT, hoặc Deepgram Nova-3 (API). Bản phiên âm tạm thời + bản cuối.
4. **LLM có gọi công cụ.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. JSON schema cho công cụ. Truyền token theo luồng.
5. **TTS theo luồng (Bài 7).** Kokoro-82M (mã nguồn mở, nhanh nhất) hoặc Cartesia Sonic (thương mại). Bắt đầu TTS sau 20 token của LLM.
6. **Phát lại.** Ra loa; mã hoá opus cho mạng băng thông thấp.
7. **Bộ xử lý ngắt lời.** Nếu VAD kích hoạt trong lúc TTS đang phát, dừng phát, huỷ LLM, khởi động lại STT.

### Ba kiểu lỗi bạn sẽ gặp

1. **Mất từ đầu.** VAD bắt đầu hơi muộn. Tiếng "hey" của người dùng bị mất. Hạ ngưỡng khởi động về 0.3 thay vì 0.5.
2. **Nhầm lẫn khi bị ngắt giữa câu trả lời.** LLM vẫn sinh tiếp sau khi người dùng ngắt lời; trợ lý nói chồng lên người dùng. Nối VAD → huỷ LLM.
3. **Bịa khi im lặng.** Whisper xuất ra "Thanks for watching" trên những khung khởi động im lặng. Luôn chặn bằng VAD.

### Các stack tham chiếu production năm 2026

| Stack | Độ trễ | Giấy phép | Ghi chú |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | API thương mại | Mặc định của ngành năm 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | phần lớn mở | Thân thiện cho tự làm |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Một mô hình duy nhất; kiến trúc khác, bài 15 |
| Vapi / Retell (quản lý sẵn) | 300-500 ms | thương mại | Ra mắt nhanh nhất; ít khả năng tuỳ biến |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | mở | Riêng tư / biên |

```figure
v4-voice-latency
```

## Tự tay làm

### Bước 1: thu âm từ mic kèm chia khối (mã giả)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Bước 2: thu một lượt nói có chặn bằng VAD

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Bước 3: STT theo luồng → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Bước 4: gọi hàm trong vòng lặp LLM

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Bước 5: xử lý ngắt lời

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Áp dụng

Xem `code/main.py` để có một mô phỏng chạy được, nối cả bảy thành phần bằng các mô hình giả lập, nhờ đó bạn thấy được hình dạng của pipeline (quy trình xử lý) ngay cả khi không có phần cứng. Với bản triển khai thật, thay các stub bằng:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` hoặc `openai-whisper`
- `openai` (`gpt-4o`) hoặc `anthropic`
- `kokoro` hoặc `cartesia`
- `sounddevice` cho I/O

## Cạm bẫy

- **Lưu log PII mãi mãi.** Âm thanh của cả lượt nói là PII ở phần lớn khu vực pháp lý. Giữ 30 ngày, mã hoá khi lưu trữ.
- **Không có barge-in.** Người dùng sẽ ngắt lời. Trợ lý của bạn phải dừng nói.
- **TTS chặn.** TTS đồng bộ chặn event loop. Dùng async hoặc một luồng riêng.
- **Không xử lý lỗi khi gọi công cụ.** Công cụ có thể lỗi. LLM phải nhận lại lỗi + thử lại một lần, rồi suy giảm một cách nhẹ nhàng.
- **Bộ lọc bịa quá tay.** Lọc quá tay thì trợ lý lặp lại "Tôi không giúp được việc đó." Lọc quá lỏng thì nó nói bất cứ điều gì. Hiệu chỉnh trên một tập giữ riêng.
- **Không có tuỳ chọn wake word (từ đánh thức).** Luôn nghe là một rủi ro về quyền riêng tư. Thêm cổng từ đánh thức (Porcupine hoặc openWakeWord).

## Đóng gói

Lưu thành `outputs/skill-voice-assistant-architect.md`. Cho trước ngân sách + quy mô + ngôn ngữ + ràng buộc tuân thủ, tạo ra một đặc tả stack đầy đủ.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó mô phỏng một lượt nói đầy đủ end-to-end với các mô-đun giả lập và in độ trễ từng tầng.
2. **Trung bình.** Thay stub STT bằng một mô hình Whisper thật trên một file `.wav` đã ghi sẵn. Đo WER và độ trễ end-to-end.
3. **Khó.** Thêm gọi hàm: triển khai `get_weather` (API bất kỳ) và `set_timer`. Định tuyến LLM qua các công cụ và kiểm chứng rằng khi người dùng nói "đặt hẹn giờ 5 phút" thì đúng hàm được kích hoạt và câu trả lời bằng giọng nói xác nhận điều đó.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Turn | Một lượt người dùng + trợ lý | Một đoạn giọng nói của người dùng được VAD bao quanh + một câu trả lời LLM-TTS. |
| Barge-in | Ngắt lời | Người dùng nói trong khi trợ lý đang nói; trợ lý dừng lại. |
| Wake word | "Hey assistant" | Bộ phát hiện từ khoá ngắn; Porcupine, Snowboy, openWakeWord. |
| End-pointing | Kết thúc lượt | Quyết định dựa trên VAD + thời gian im lặng tối thiểu rằng người dùng đã nói xong. |
| Pre-roll | Bộ đệm trước giọng nói | Giữ 200-400 ms âm thanh trước khi VAD kích hoạt để tránh mất từ đầu. |
| Tool call | Gọi hàm | LLM phát ra JSON; runtime điều phối; kết quả được đưa lại trong vòng lặp. |

## Đọc thêm

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — tham chiếu chuẩn production.
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — framework thân thiện cho tự làm.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — con đường quản lý sẵn, thuần giọng nói.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — tham chiếu full-duplex (Bài 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — cổng từ đánh thức.
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — gọi hàm của LLM.

---

[← Bài trước: Xử lý âm thanh trong thời gian thực](../../11-real-time-audio-processing/docs/vi.md) · [Bài sau: Neural Audio Codecs →](../../13-neural-audio-codecs/docs/vi.md)
