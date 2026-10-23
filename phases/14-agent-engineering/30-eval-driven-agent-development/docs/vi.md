# Phát triển agent dựa trên đánh giá

> Anthropic khuyên: "bắt đầu với prompt đơn giản, tối ưu bằng đánh giá toàn diện, và chỉ thêm hệ thống agent nhiều bước khi cần." Đánh giá không phải bước cuối; đó là vòng lặp chi phối mọi lựa chọn khác trong Giai đoạn 14.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Toàn bộ Giai đoạn 14.
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu ba lớp đánh giá — benchmark tĩnh (bộ kiểm chuẩn cố định), đánh giá offline tùy chỉnh, đánh giá online trong môi trường thực tế — và mục đích từng lớp.
- Giải thích vòng lặp chặt chẽ evaluator-optimizer (đánh giá và tối ưu).
- Mô tả cách làm tốt nhất năm 2026: bài đánh giá nằm cạnh code, chạy trong CI, làm điều kiện thông qua PR.
- Liên hệ mỗi bài ở Giai đoạn 14 với trường hợp đánh giá mà bài đó tạo ra.

## Vấn đề

Agent vượt qua demo nhưng thất bại trong thực tế theo cách demo không dự đoán được. Benchmark trả lời "mô hình này có năng lực rộng không?", chứ không phải "agent này có tạo đúng bản vá cho sản phẩm của tôi không?" Giải pháp: đánh giá liên tục ở ba lớp, gắn mỗi quy tắc bảo vệ và quy tắc đã học với một trường hợp đánh giá.

## Khái niệm

### Ba lớp đánh giá

1. **Benchmark tĩnh** — SWE-bench Verified cho code (Bài 19), WebArena/OSWorld cho duyệt web / máy tính (Bài 20), GAIA cho năng lực tổng quát (Bài 19), BFCL V4 cho sử dụng công cụ (Bài 06). Dùng để so sánh mô hình và chặn suy giảm chất lượng. Rò rỉ lời giải là có thật: SWE-bench+ phát hiện 32.67% lời giải bị lộ. Luôn báo cáo điểm Verified / đã kiểm tra theo +.

2. **Đánh giá offline tùy chỉnh** — theo đặc thù sản phẩm:
   - LLM-as-judge (LLM làm giám khảo; Langfuse, Phoenix, Opik — Bài 24).
   - Dựa trên thực thi (chạy bản vá, kiểm tra test).
   - Dựa trên chuỗi hành động (so trình tự hành động với mẫu chuẩn; OSWorld-Human cho thấy agent hàng đầu dùng số bước gấp 1.4-2.7x mẫu chuẩn).

3. **Đánh giá online** — trong thực tế:
   - Phát lại phiên (Langfuse).
   - Cảnh báo khi quy tắc bảo vệ kích hoạt (Bài 16, 21).
   - Theo dõi chi phí / độ trễ từng bước (span OTel ở Bài 23).

### Evaluator-optimizer (Anthropic)

Vòng lặp chặt chẽ:

1. Bộ đề xuất tạo đầu ra.
2. Bộ đánh giá nhận xét.
3. Tinh chỉnh đến khi bộ đánh giá chấp nhận.

Đây là Self-Refine (Bài 05) được khái quát hóa. Bạn có thể bọc mọi luồng agent quan trọng trong evaluator-optimizer để tăng độ tin cậy.

### Cách làm tốt nhất năm 2026

- Bài đánh giá nằm cạnh code.
- Chạy trong CI trên mỗi PR.
- Chỉ gộp code khi điểm đánh giá đạt yêu cầu (ví dụ: "không giảm quá 5% so với main").
- Mỗi quy tắc bảo vệ gắn với một trường hợp đánh giá.
- Mỗi quy tắc đã học (Reflexion, pro-workflow learn-rule) gắn với một trường hợp thất bại.

### Kết nối các bài học Giai đoạn 14

Mỗi bài ở Giai đoạn 14 tạo ra các trường hợp đánh giá:

| Bài học | Trường hợp đánh giá được tạo ra |
|--------|------------------------|
| 01 Vòng lặp agent | Hết ngân sách, chặn vòng lặp vô hạn |
| 02 ReWOO | Bộ lập kế hoạch lập lại kế hoạch đúng khi công cụ lỗi |
| 03 Reflexion | Bài học rút ra được áp dụng khi thử lại |
| 05 Self-Refine/CRITIC | Giám khảo chấp nhận đầu ra đã tinh chỉnh |
| 06 Sử dụng công cụ | Chuyển kiểu đối số thành công; từ chối công cụ không xác định |
| 07-10 Bộ nhớ | Trích dẫn từ kết quả truy xuất khớp nguồn; vô hiệu hóa thông tin cũ |
| 12 Các mẫu workflow | Mỗi mẫu tạo đầu ra đúng |
| 13 LangGraph | Tiếp tục chạy tái tạo chính xác trạng thái |
| 14 Các actor AutoGen | DLQ tiếp nhận handler bị lỗi |
| 16 OpenAI Agents SDK | Quy tắc bảo vệ kích hoạt với đúng đầu vào |
| 17 Claude Agent SDK | Kết quả subagent trả về bộ điều phối |
| 19-20 Benchmark | Điểm SWE-bench Verified, tỷ lệ thành công WebArena, hiệu quả OSWorld |
| 21 Sử dụng máy tính | Kiểm tra an toàn từng bước phát hiện DOM bị chèn nội dung độc hại |
| 23 OTel | Span phát ra các thuộc tính bắt buộc |
| 26 Các dạng thất bại | Bộ phát hiện gắn nhãn thất bại đã biết |
| 27 Chèn lệnh vào prompt | PVE từ chối kết quả truy xuất bị đầu độc |
| 28 Điều phối | Bộ giám sát chuyển việc đến đúng chuyên gia |
| 29 Các dạng runtime | DLQ xử lý tỷ lệ thất bại N% |

Nếu bộ đánh giá có trường hợp cho từng mục, bạn đã bao phủ Giai đoạn 14.

### Khi phát triển dựa trên đánh giá thất bại

- **Không có baseline (mốc tham chiếu).** Kết quả đánh giá khó hiểu nếu thiếu phiên bản tốt gần nhất để đối chiếu. Hãy lưu baseline.
- **LLM làm giám khảo thiếu căn cứ.** Giám khảo cũng có thể bịa. Theo mẫu CRITIC (Bài 05), giám khảo kiểm chứng bằng công cụ bên ngoài.
- **Quá khớp với bài đánh giá.** Tối ưu cho bài đánh giá có thể làm giảm tính hữu ích trong thực tế. Hãy luân phiên thay đổi các trường hợp.
- **Bài đánh giá thiếu ổn định.** Trường hợp không tất định gây báo động giả. Hãy cố định seed, lưu ảnh chụp trạng thái.

```figure
ae-eval-three-layers
```

## Tự tay làm

`code/main.py` là bộ đánh giá dùng stdlib:

- Danh mục trường hợp theo loại (benchmark, tùy chỉnh, online).
- Agent chạy theo kịch bản để kiểm thử.
- Vòng lặp evaluator-optimizer: đề xuất, đánh giá, tinh chỉnh đến khi đạt hoặc hết số vòng tối đa.
- Điều kiện thông qua CI: tổng hợp tỷ lệ đạt và so mức suy giảm với baseline.

Chạy:

```
python3 code/main.py
```

Đầu ra: đạt/không đạt từng trường hợp, cờ báo suy giảm, quyết định thông qua CI.

## Áp dụng

- Viết trường hợp đánh giá trong cùng repo với code agent.
- Chạy trên mỗi PR qua CI.
- Cho build thất bại nếu chất lượng giảm.
- Theo dõi tỷ lệ đạt theo thời gian.
- Gắn mỗi thất bại trong thực tế với một trường hợp mới.

## Đóng gói

`outputs/skill-eval-suite.md` xây dựng bộ đánh giá ba lớp cho sản phẩm agent, có điều kiện thông qua CI và theo dõi suy giảm.

## Bài tập

1. Chọn một thất bại thực tế của bạn. Viết trường hợp đánh giá tái hiện nó. Agent hiện vượt qua được không?
2. Xây dựng bộ tiêu chí LLM làm giám khảo cho lĩnh vực của bạn, gồm ba khía cạnh (tính chính xác, giọng điệu, phạm vi). Chấm 50 phiên.
3. Tích hợp bộ đánh giá vào CI. Cho build thất bại nếu chất lượng giảm >=5%.
4. Thêm chỉ số hiệu quả chuỗi hành động: agent dùng bao nhiêu bước so với chuỗi mẫu chuẩn?
5. Gắn mỗi bài Giai đoạn 14 với một trường hợp trong bộ đánh giá. Còn thiếu bài nào? Đó là khoảng trống cần khắc phục.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Static benchmark | "Bài đánh giá có sẵn" | SWE-bench, GAIA, AgentBench, WebArena, OSWorld |
| Custom offline eval | "Đánh giá theo lĩnh vực" | LLM làm giám khảo / thực thi / chuỗi hành động theo đặc thù sản phẩm |
| Online eval | "Đánh giá trong thực tế" | Phát lại phiên, cảnh báo từ quy tắc bảo vệ, theo dõi chi phí/độ trễ |
| Evaluator-optimizer | "Đề xuất-đánh giá-tinh chỉnh" | Lặp đến khi giám khảo chấp nhận |
| CI gate | "Điều kiện chặn gộp code" | Cho build thất bại khi kết quả đánh giá giảm |
| Baseline | "Phiên bản tốt gần nhất" | Điểm tham chiếu để phát hiện suy giảm |
| Trajectory efficiency | "Số bước so với mẫu chuẩn" | Số bước của agent chia cho số bước tối thiểu của chuyên gia |

## Đọc thêm

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — "bắt đầu đơn giản, tối ưu bằng đánh giá"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — benchmark đã tuyển chọn
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — benchmark sử dụng công cụ
- [Tài liệu Langfuse](https://langfuse.com/) — đánh giá và phát lại phiên trong thực tế

---

[← Bài trước: Runtime production: Hàng đợi, Sự kiện, Cron](../../29-production-runtimes/docs/vi.md) · [Bài sau: Kỹ thuật workbench cho agent: Vì sao model giỏi vẫn thất bại →](../../31-agent-workbench-why-models-fail/docs/vi.md)
