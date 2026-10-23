# Sử dụng công cụ và function calling (gọi hàm)

> Toolformer (Schick et al., 2023) đã khởi đầu việc tự chú thích công cụ theo kiểu tự giám sát. Berkeley Function Calling Leaderboard V4 (Patil et al., 2025) đặt mốc 2026: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination. Lượt đơn đã được giải quyết. Bộ nhớ, ra quyết định động, và chuỗi công cụ tầm xa thì chưa.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent), Giai đoạn 13, Bài 01 (đi sâu vào function calling)
**Time:** ~60 phút

## Mục tiêu học tập

- Giải thích tín hiệu huấn luyện tự giám sát của Toolformer: chỉ giữ chú thích công cụ khi việc thực thi làm giảm loss của token kế tiếp.
- Kể tên năm nhóm đánh giá của BFCL V4 và mỗi nhóm đo lường điều gì.
- Cài đặt một registry công cụ thuần stdlib với kiểm tra schema, ép kiểu tham số, và cô lập sandbox khi thực thi.
- Chẩn đoán ba vấn đề còn mở của 2026: chuỗi công cụ tầm xa, ra quyết định động, và bộ nhớ.

## Vấn đề

Sử dụng công cụ thời kỳ đầu hỏi rằng: mô hình có dự đoán được một lệnh gọi hàm đúng không? Sử dụng công cụ hiện đại hỏi rằng: mô hình có nối được công cụ qua 40 bước, có bộ nhớ, có quan sát một phần, có phục hồi sau lỗi công cụ, mà không bịa ra những công cụ không tồn tại không?

Toolformer đã dựng nên đường cơ sở: mô hình có thể học khi nào nên gọi công cụ bằng tự giám sát. BFCL V4 định nghĩa mục tiêu đánh giá của 2026. Khoảng cách giữa hai thứ đó chính là nơi các agent trong sản phẩm thật đang sống.

## Khái niệm

### Toolformer (Schick et al., NeurIPS 2023)

Ý tưởng: để mô hình tự chú thích kho ngữ liệu tiền huấn luyện của nó bằng các lệnh gọi API ứng viên. Với mỗi ứng viên, thực thi nó. Chỉ giữ chú thích nếu việc đưa kết quả công cụ (tool result) vào làm giảm loss ở token kế tiếp. Tinh chỉnh trên kho ngữ liệu đã lọc.

Các công cụ được dùng: máy tính, hệ thống QA, công cụ tìm kiếm, dịch giả, lịch. Tín hiệu tự giám sát hoàn toàn nói về việc công cụ có giúp dự đoán văn bản hay không — không có nhãn của con người.

Kết quả về quy mô: sử dụng công cụ xuất hiện theo quy mô. Các mô hình nhỏ bị hại bởi chú thích công cụ; mô hình lớn thì được lợi. Đây là lý do các mô hình frontier 2026 có sẵn khả năng dùng công cụ mạnh, trong khi hầu hết mô hình 7B cần tinh chỉnh rõ ràng cho sử dụng công cụ mới đáng tin.

### Berkeley Function Calling Leaderboard V4 (Patil et al., ICML 2025)

BFCL là đánh giá thực tế mặc định của 2026. Thành phần của V4:

- **Agentic (40%)** — toàn bộ hành trình của agent: bộ nhớ, nhiều lượt, quyết định động.
- **Multi-Turn (30%)** — hội thoại tương tác với chuỗi công cụ.
- **Live (10%)** — prompt thật do người dùng gửi lên (phân phối khó hơn).
- **Non-Live (10%)** — các ca kiểm thử tổng hợp.
- **Hallucination (10%)** — phát hiện khi nào không nên gọi công cụ nào.

V3 giới thiệu đánh giá dựa trên trạng thái: sau một chuỗi công cụ, kiểm tra trạng thái thật của API (ví dụ "file đã được tạo chưa?") thay vì so khớp AST của các lệnh gọi công cụ. V4 thêm các nhóm web search, memory, và độ nhạy định dạng.

Phát hiện chính của 2026: gọi hàm một lượt gần như đã được giải quyết. Thất bại tập trung ở bộ nhớ (mang ngữ cảnh qua các lượt), ra quyết định động (chọn công cụ dựa trên kết quả trước đó), chuỗi tầm xa (lệch hướng sau hơn 20 bước), và phát hiện ảo giác (từ chối gọi khi không công cụ nào phù hợp).

### Tool schema (lược đồ công cụ)

Mọi nhà cung cấp đều có một schema. Chúng khác nhau ở chi tiết nhưng cùng một hình dạng:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic dùng trực tiếp `input_schema`. OpenAI dùng `function.parameters`. Cả hai đều nhận JSON Schema. Phần mô tả có sức nặng — mô hình đọc chúng để chọn đúng công cụ. Mô tả công cụ tồi là nguyên nhân gốc số 1 của các thất bại chọn nhầm công cụ.

### Kiểm tra tham số

Đừng tin bất kỳ lệnh gọi công cụ nào. Hãy kiểm tra:

1. **Ép kiểu.** Mô hình có thể trả về chuỗi "5" trong khi schema nói int. Ép kiểu nếu không mơ hồ; từ chối nếu mơ hồ.
2. **Kiểm tra enum.** Nếu schema nói `status in {"open", "closed"}` mà mô hình phát ra `"in_progress"`, hãy từ chối kèm lỗi mô tả rõ.
3. **Trường bắt buộc.** Thiếu trường bắt buộc -> trả ngay một quan sát lỗi về cho mô hình, không phải crash.
4. **Kiểm tra định dạng.** Ngày, email, URL — kiểm tra bằng parser thật, không dùng regex.

Mọi thất bại khi kiểm tra nên trả về một quan sát có cấu trúc để mô hình có thể thử lại với đúng hình dạng.

### Gọi công cụ song song

Các nhà cung cấp hiện đại hỗ trợ gọi công cụ song song trong một lượt của assistant. Vòng lặp:

1. Mô hình phát ra 3 lệnh gọi công cụ với các `tool_use_id` khác nhau.
2. Runtime thực thi chúng (song song nếu độc lập).
3. Mỗi kết quả quay lại dưới dạng một khối `tool_result` được ghép tương ứng qua `tool_use_id`.

Nguyên tắc kỹ thuật: coi các ID tương ứng là có sức nặng. Hoán đổi chúng là bạn nhận về việc định tuyến sai công cụ với sai kết quả.

### Cô lập sandbox

Việc thực thi công cụ chính là ranh giới sandbox. Xem Bài 09 để biết chi tiết. Bản ngắn: mỗi công cụ nên nêu rõ bề mặt đọc/ghi, truy cập mạng, timeout, giới hạn bộ nhớ. Một `run_shell(cmd)` chung chung là cờ đỏ; một `git_status()` cụ thể thì an toàn hơn.

```figure
tool-routing
```

## Tự tay làm

`code/main.py` cài đặt một registry công cụ đúng hình dạng của sản phẩm:

- Bộ kiểm tra tập con JSON Schema (chỉ stdlib).
- Đăng ký công cụ kèm mô tả, input schema, timeout, và executor.
- Ép kiểu tham số và kiểm tra enum.
- Điều phối công cụ song song với các ID tương ứng.
- Quan sát lỗi dưới dạng chuỗi có cấu trúc.

Chạy thử:

```
python3 code/main.py
```

Trace cho thấy một mini agent gọi ba công cụ trong một lượt, với một lệnh gọi cố tình sai định dạng bị từ chối kèm lỗi mô tả rõ để mô hình có thể xử lý tiếp.

## Áp dụng

Mỗi nhà cung cấp có schema công cụ riêng — Anthropic, OpenAI, Gemini, Bedrock. Dùng một lớp chuyển đổi (OpenAI Agents SDK, Vercel AI SDK, LangChain tool adapter) nếu bạn cần đa nhà cung cấp. BFCL là benchmark tham chiếu — hãy chạy nó với agent của bạn trước khi ship nếu sử dụng công cụ là phần cốt lõi của sản phẩm.

## Đóng gói

`outputs/skill-tool-registry.md` sinh ra một catalog công cụ, schema, và registry cho một miền nhiệm vụ cho trước. Bao gồm cả kiểm tra chất lượng mô tả (mô tả của mỗi công cụ có nói cho mô hình biết khi nào nên dùng nó không?).

## Bài tập

1. Thêm một công cụ "no-op" cho phép mô hình từ chối rõ ràng việc dùng bất kỳ công cụ nào khác. Đo trên một bài kiểm thử ảo giác kiểu BFCL.
2. Cài đặt ép kiểu tham số cho int-dạng-chuỗi và float-dạng-chuỗi. Ép kiểu bắt đầu che giấu lỗi thật từ đâu?
3. Thêm timeout theo từng công cụ và một circuit breaker (từ chối công cụ trong 60s sau 3 lần thất bại liên tiếp). Điều này thay đổi gì về cách mô hình phục hồi?
4. Đọc mô tả BFCL V4. Chọn một nhóm (ví dụ "multi-turn") và chạy 10 prompt mẫu qua agent của bạn. Báo cáo tỉ lệ pass.
5. Chuyển bộ kiểm tra stdlib sang Pydantic hoặc Zod. Pydantic/Zod bắt được gì mà bản đồ chơi bỏ sót?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Function calling | "Sử dụng công cụ" | Gọi công cụ dưới dạng đầu ra có cấu trúc với schema đã kiểm tra |
| Toolformer | "Tự chú thích công cụ tự giám sát" | Schick 2023 — giữ lại những lệnh gọi công cụ mà kết quả của chúng làm giảm loss token kế tiếp |
| BFCL | "Berkeley Function Calling Leaderboard" | Benchmark 2026: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "Chữ ký hàm cho mô hình" | name, description, JSON Schema của các tham số |
| tool_use_id | "ID tương ứng" | Gắn một lệnh gọi công cụ với kết quả của nó; thiết yếu cho điều phối song song |
| Hallucination detection | "Biết khi nào không nên gọi" | Nhóm của V4: từ chối gọi khi không công cụ nào phù hợp |
| Argument coercion | "Sửa chuỗi thành int" | Các bản sửa hẹp cho lệch schema có thể đoán trước; từ chối nếu mơ hồ |
| Sandboxing | "Ranh giới thực thi công cụ" | Bề mặt đọc/ghi, mạng, timeout, giới hạn bộ nhớ theo từng công cụ |

## Đọc thêm

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — tự chú thích công cụ tự giám sát
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — benchmark đánh giá của 2026
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — tool schema trong sản phẩm ở Claude Agent SDK
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — kiểu function tool và Guardrails
