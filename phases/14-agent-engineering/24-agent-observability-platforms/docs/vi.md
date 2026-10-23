# Khả năng quan sát agent: Langfuse, Phoenix, Opik

> Ba nền tảng quan sát agent mã nguồn mở chiếm ưu thế năm 2026. Langfuse (MIT) — hơn 6M lượt cài đặt/tháng, tracing (vết) + quản lý prompt + eval + session replay (phát lại phiên). Arize Phoenix (Elastic 2.0) — eval chuyên sâu riêng cho agent, độ liên quan RAG, tự động instrumentation OpenInference. Comet Opik (Apache 2.0) — tối ưu prompt tự động, guardrail, phát hiện ảo giác bằng LLM-judge.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 23 (OTel GenAI)
**Time:** ~45 phút

## Mục tiêu học tập

- Kể tên ba nền tảng quan sát (observability platform) agent mã nguồn mở hàng đầu và giấy phép của chúng.
- Phân biệt điểm mạnh nhất của từng nền tảng: Langfuse (quản lý prompt + phiên), Phoenix (RAG + tự động instrumentation), Opik (tối ưu hoá + guardrail).
- Giải thích vì sao 89% tổ chức báo cáo đã có khả năng quan sát agent vào năm 2026.
- Triển khai một pipeline trace → dashboard chỉ với stdlib, kèm đánh giá bằng LLM-judge.

## Vấn đề

OTel GenAI (Bài 23) cho bạn schema. Bạn vẫn cần nền tảng để nạp span, chạy đánh giá, lưu các phiên bản prompt và phát hiện hồi quy. Ba ứng viên mỗi bên nhấn mạnh một phần khác nhau của vòng đời.

## Khái niệm

### Langfuse (MIT)

- Hơn 6M lượt cài đặt SDK/tháng, hơn 19k sao GitHub.
- Tính năng: tracing, quản lý prompt có versioning + playground, đánh giá (LLM-as-judge, phản hồi người dùng, tuỳ chỉnh), session replay.
- Tháng 6/2025: các module trước đây là thương mại (LLM-as-a-judge, hàng đợi chú thích, thử nghiệm prompt, Playground) được mở mã nguồn theo MIT.
- Mạnh nhất ở: khả năng quan sát đầu-cuối với vòng quản lý prompt khép kín.

### Arize Phoenix (Elastic License 2.0)

- Đánh giá chuyên sâu hơn cho agent: phân cụm vết, phát hiện bất thường, độ liên quan truy xuất cho RAG.
- Tự động instrumentation OpenInference gốc.
- Kết hợp với Arize AX (bản có quản lý) cho môi trường production.
- Không có versioning prompt — được định vị là công cụ phát hiện dịch chuyển/hồi quy hành vi, dùng song song với các nền tảng rộng hơn.
- Mạnh nhất ở: độ liên quan RAG, dịch chuyển hành vi, phát hiện bất thường.

### Comet Opik (Apache 2.0)

- Tối ưu prompt tự động qua thử nghiệm A/B.
- Guardrail (che PII, ràng buộc chủ đề).
- Phát hiện ảo giác bằng LLM-judge.
- Benchmark từ chính đo lường của Comet: Opik ghi log + eval trong 23.44s so với Langfuse 327.15s (chênh ~14x) — hãy xem benchmark của nhà cung cấp chỉ mang tính định hướng.
- Mạnh nhất ở: vòng tối ưu hoá, thử nghiệm tự động, thực thi guardrail.

### Dữ liệu ngành

Theo Maxim (phân tích thực địa 2026): 89% tổ chức đã có khả năng quan sát agent; vấn đề chất lượng là rào cản lớn nhất ở production (32% người trả lời nêu lên).

### Chọn một nền tảng

| Nhu cầu | Lựa chọn |
|------|------|
| Tất cả trong một, có quản lý prompt | Langfuse |
| Đánh giá RAG sâu + phát hiện dịch chuyển | Phoenix |
| Tối ưu hoá tự động + guardrail | Opik |
| Giấy phép mở, không ELv2 | Langfuse (MIT) hoặc Opik (Apache 2.0) |
| Tích hợp Datadog / New Relic | Bất kỳ — cả ba đều xuất OTel |

### Chỗ mẫu hình này hay hỏng

- **Không có chiến lược eval.** Tracing mà không đánh giá thì chỉ là ghi log đắt tiền.
- **Tự viết LLM-judge mà không có nền kiểm chứng.** Mẫu hình CRITIC (Bài 05) vẫn áp dụng — judge cần công cụ bên ngoài để kiểm chứng sự thật.
- **Phiên bản prompt không gắn với vết.** Khi production hồi quy, bạn không thể bisect ra prompt đã gây lỗi.

```figure
wb-trace-ingest
```

## Tự tay làm

`code/main.py` triển khai một bộ thu thập vết bằng stdlib + bộ đánh giá LLM-judge:

- Nạp các span có dạng GenAI.
- Nhóm theo phiên, gắn nhãn các lần chạy thất bại (guardrail kích hoạt, eval độ tin cậy thấp).
- Một LLM-judge theo kịch bản, chấm điểm phản hồi của agent theo rubric.
- Bản tóm tắt kiểu dashboard: tỷ lệ thất bại, các lý do thất bại hàng đầu, phân phối điểm eval.

Chạy thử:

```
python3 code/main.py
```

Kết quả: điểm eval theo từng phiên và cách phân loại thất bại, khớp với những gì Langfuse/Phoenix/Opik sẽ hiển thị.

## Áp dụng

- **Langfuse** tự host hoặc cloud; nối qua OTel hoặc SDK của họ.
- **Arize Phoenix** tự host; tự động instrumentation OpenInference.
- **Comet Opik** tự host hoặc cloud; vòng tối ưu hoá tự động.
- **Datadog LLM Observability** cho đội vừa ops vừa ML đã dùng Datadog.

## Đóng gói

`outputs/skill-obs-platform-wiring.md` giúp chọn một nền tảng rồi nối vết + eval + phiên bản prompt vào một agent đã có.

## Bài tập

1. Xuất một tuần vết OTel lên Langfuse cloud (bậc miễn phí). Phiên nào thất bại? Vì sao?
2. Viết rubric LLM-judge cho lĩnh vực của bạn (đúng sự thật, giọng điệu, bám phạm vi). Thử trên 50 vết.
3. So sánh versioning prompt của Langfuse với phân cụm vết của Phoenix. Cái nào cho bạn biết nhanh hơn cái gì đã hỏng?
4. Đọc tài liệu guardrail của Opik. Nối một guardrail che PII vào một lần chạy agent của bạn.
5. Benchmark ba nền tảng trên corpus của bạn. Bỏ qua con số nhà cung cấp công bố; tự đo lấy.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Tracing | "Bộ thu span" | Nạp span từ OTel / SDK; đánh chỉ mục theo phiên |
| Prompt management | "CMS cho prompt" | Prompt được đánh phiên bản và gắn với vết |
| LLM-as-judge | "Eval tự động" | Một LLM riêng chấm đầu ra của agent theo rubric |
| Session replay | "Phát lại vết" | Bước qua các lần chạy trước để gỡ lỗi |
| RAG relevancy | "Chất lượng truy xuất" | Ngữ cảnh truy xuất được có khớp với truy vấn không |
| Trace clustering | "Gom nhóm hành vi" | Gom các lần chạy tương tự để phát hiện dịch chuyển |
| Guardrail enforcement | "Chính sách ngay lúc ghi log" | Kiểm tra PII/độc hại/phạm vi trên nội dung được ghi log |

## Đọc thêm

- [Tài liệu Langfuse](https://langfuse.com/) — tracing, eval, quản lý prompt
- [Tài liệu Arize Phoenix](https://docs.arize.com/phoenix) — tự động instrumentation, phát hiện dịch chuyển
- [Comet Opik](https://www.comet.com/site/products/opik/) — tối ưu hoá + guardrail
- [Quy ước ngữ nghĩa OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — schema mà cả ba đều dùng

---

[← Bài trước: Quy ước ngữ nghĩa GenAI của OpenTelemetry](../../23-otel-genai-conventions/docs/vi.md) · [Bài sau: Tranh luận và cộng tác đa tác nhân →](../../25-multi-agent-debate/docs/vi.md)
