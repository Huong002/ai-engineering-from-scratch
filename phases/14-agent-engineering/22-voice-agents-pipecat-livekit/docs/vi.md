# Agent thoại: Pipecat và LiveKit

> Agent thoại (voice agent) là một hạng mục sản phẩm hạng nhất trong năm 2026. Pipecat cho bạn một pipeline dựa trên frame bằng Python, gồm VAD (phát hiện hoạt động giọng nói) → STT → LLM → TTS → transport. LiveKit Agents nối các model AI tới người dùng qua WebRTC. Mục tiêu độ trễ khi chạy thật rơi vào 450–600ms end-to-end cho các stack cao cấp.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent), Giai đoạn 14, Bài 12 (mẫu workflow)
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả pipeline dựa trên frame của Pipecat: DOWNSTREAM (source→sink) và UPSTREAM (điều khiển).
- Kể tên các tầng chuẩn của pipeline thoại và những transport mà Pipecat hỗ trợ.
- Giải thích hai lớp voice agent của LiveKit Agents (MultimodalAgent, VoicePipelineAgent) và khi nào dùng cái nào.
- Tóm tắt kỳ vọng độ trễ khi chạy thật năm 2026 và cách chúng định hình lựa chọn kiến trúc.

## Vấn đề

Agent thoại không phải một vòng lặp văn bản gắn thêm TTS. Ngân sách độ trễ (latency budget) rất khắc nghiệt (~600ms), âm thanh một phần là mặc định, turn detection (phát hiện lượt nói) là một model, và transport trải từ thoại SIP tới WebRTC. Hoặc bạn dựng pipeline dựa trên frame (Pipecat), hoặc bạn dựa vào nền tảng (LiveKit).

## Khái niệm

### Pipecat (pipecat-ai/pipecat)

- Framework pipeline dựa trên frame bằng Python.
- Chuỗi `Frame` → `FrameProcessor`.
- Hai hướng luồng:
  - **DOWNSTREAM** — source → sink (âm thanh vào, TTS ra).
  - **UPSTREAM** — phản hồi và điều khiển: huỷ, số đo, barge-in (ngắt lời).
- `PipelineTask` quản lý vòng đời với các event (`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`) và các observer cho số đo/tracing/RTVI.

Pipeline điển hình:

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Các transport: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows thêm hội thoại có cấu trúc (máy trạng thái). Pipecat Cloud là runtime được quản lý.

### LiveKit Agents (livekit/agents)

- Nối các model AI tới người dùng qua WebRTC.
- Khái niệm chính: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- Hai lớp voice agent:
  - **MultimodalAgent** — âm thanh trực tiếp qua OpenAI Realtime hoặc tương đương.
  - **VoicePipelineAgent** — thác STT → LLM → TTS; cho phép điều khiển ở mức văn bản.
- Phát hiện lượt nói theo ngữ nghĩa bằng một model transformer.
- Tích hợp MCP sẵn có.
- Thoại qua SIP.
- Hơn 50 model không cần API key qua LiveKit Inference; hơn 200 model nữa qua plugin.

### Nền tảng thương mại

Vapi (~450–600ms trên một stack cao cấp đã tối ưu) và Retell (~600ms end-to-end qua 180 cuộc gọi thử) được dựng trên những thứ này. Chọn một nền tảng khi bạn muốn một stack thoại được quản lý mà không cần đội WebRTC.

### Chỗ mẫu hình này hay hỏng

- **Không xử lý barge-in.** Người dùng ngắt lời; agent vẫn nói tiếp. Cần frame huỷ UPSTREAM trong Pipecat, tương đương trong LiveKit.
- **Bỏ qua độ tin cậy của STT.** Bản ghi có độ tin cậy thấp vẫn được đưa vào LLM như chân lý. Hãy chặn theo độ tin cậy hoặc yêu cầu xác nhận.
- **TTS bị cắt giữa câu.** Khi pipeline huỷ giữa một phát ngôn, TTS cần biết hoặc cắt âm thanh.
- **Bỏ qua ngân sách độ trễ.** Mỗi thành phần thêm 50–200ms. Hãy cộng cả chuỗi trước khi phát hành.

### Độ trễ điển hình năm 2026

- VAD: 20–60ms
- STT partial: 100–250ms
- LLM first token: 150–400ms
- TTS first audio: 100–200ms
- Transport RTT: 30–80ms

450–600ms end-to-end là cao cấp. 800–1200ms là phổ biến. Bất cứ gì > 1500ms đều nghe như hỏng.

```figure
voice-pipeline
```

## Tự tay làm

`code/main.py` là một pipeline đồ chơi dựa trên frame với:

- Các kiểu `Frame` (audio, transcript, text, tts_audio, control).
- Giao diện `Processor` với `process(frame)`.
- Một pipeline năm tầng (VAD → STT → LLM → TTS → transport) dưới dạng các processor được kịch bản hoá.
- Một frame huỷ UPSTREAM để minh hoạ barge-in.

Chạy nó:

```
python3 code/main.py
```

Bản trace hiển thị luồng bình thường và một lần huỷ barge-in dừng TTS giữa phát ngôn.

## Áp dụng

- **Pipecat** để toàn quyền kiểm soát — processor tuỳ biến, ưu tiên Python, provider cắm được.
- **LiveKit Agents** cho triển khai ưu tiên WebRTC và thoại.
- **Vapi / Retell** cho agent thoại dạng hosted mà không cần đội WebRTC.
- **OpenAI Realtime / Gemini Live** cho âm thanh vào/âm thanh ra trực tiếp (MultimodalAgent).

## Đóng gói

`outputs/skill-voice-pipeline.md` dựng khung một pipeline thoại kiểu Pipecat với VAD + STT + LLM + TTS + transport cùng xử lý barge-in.

## Bài tập

1. Thêm một observer số đo vào pipeline đồ chơi của bạn: đếm số frame mỗi tầng mỗi giây. Độ trễ tích tụ ở đâu?
2. Cài STT chặn theo độ tin cậy: dưới ngưỡng thì hỏi "bạn nhắc lại giúp được không?"
3. Thêm phát hiện lượt nói theo ngữ nghĩa: quy tắc đơn giản — nếu bản ghi kết thúc bằng "?", hết lượt.
4. Đọc tài liệu transport của Pipecat. Thay transport stdlib bằng cấu hình SmallWebRTCTransport (bản khung).
5. Đo một OpenAI Realtime so với thác STT+LLM+TTS trên cùng một truy vấn. Điều khiển ở mức văn bản phải trả giá độ trễ bao nhiêu?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Frame | "Sự kiện" | Đơn vị dữ liệu có kiểu trong pipeline (audio, transcript, text, control) |
| Processor | "Tầng pipeline" | Handler có process(frame) |
| DOWNSTREAM | "Luồng xuôi" | Từ source tới sink: âm thanh vào, giọng nói ra |
| UPSTREAM | "Luồng phản hồi" | Điều khiển: huỷ, số đo, barge-in |
| VAD | "Voice activity detection" | Phát hiện khi người dùng đang nói |
| Semantic turn detection | "Kết thúc lượt thông minh" | Quyết định dựa trên model rằng người dùng đã nói xong |
| MultimodalAgent | "Agent âm thanh trực tiếp" | Âm thanh vào, âm thanh ra; không có văn bản ở giữa |
| VoicePipelineAgent | "Agent thác" | STT + LLM + TTS; điều khiển ở mức văn bản |

## Đọc thêm

- [Tài liệu Pipecat](https://docs.pipecat.ai/getting-started/introduction) — pipeline dựa trên frame, processor, transport
- [Tài liệu LiveKit Agents](https://docs.livekit.io/agents/) — WebRTC + các primitive thoại
- [Vapi](https://vapi.ai/) — nền tảng thoại được quản lý
- [Retell AI](https://www.retellai.com/) — thoại được quản lý, có benchmark độ trễ

---

[← Bài trước: Computer use (dùng máy tính): Claude, OpenAI CUA, Gemini](../../21-computer-use-agents/docs/vi.md) · [Bài sau: Quy ước ngữ nghĩa GenAI của OpenTelemetry →](../../23-otel-genai-conventions/docs/vi.md)
