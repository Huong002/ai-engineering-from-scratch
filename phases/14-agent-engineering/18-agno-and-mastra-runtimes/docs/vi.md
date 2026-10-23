# Runtime cho agent production — Khởi tạo tức thì và Workflow có kiểu

> Một runtime (môi trường chạy) cho agent production tối ưu hoá đúng những thứ mà framework prototyping bỏ qua: chi phí khởi tạo, bề mặt workflow có kiểu, và một backend sẵn sàng phục vụ. Cặp đôi của năm 2026: Agno (Python) nhắm tới khởi tạo agent trong hàng microsecond và backend FastAPI không trạng thái. Mastra mang tới agent, tool, workflow, định tuyến model hợp nhất và lưu trữ tổng hợp trên nền Vercel AI SDK.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent), Giai đoạn 14, Bài 13 (LangGraph)
**Time:** ~45 phút

## Mục tiêu học tập

- Xác định các mục tiêu hiệu năng của Agno và khi nào chúng thực sự quan trọng.
- Kể tên ba primitive của Mastra — Agents, Tools, Workflows (quy trình) — và các server adapter được hỗ trợ.
- Giải thích vì sao backend FastAPI không trạng thái, phạm vi theo phiên, là con đường production được khuyến nghị cho Agno.
- Chọn Agno hay Mastra cho một stack cụ thể (ưu tiên Python hay ưu tiên TypeScript).

## Vấn đề

LangGraph, AutoGen, CrewAI đều nặng về framework. Những nhóm chỉ muốn "đúng vòng lặp agent, nhanh, ngay trong runtime của tôi" sẽ tìm tới Agno (Python) hoặc Mastra (TypeScript). Cả hai đánh đổi một phần primitive do framework sở hữu để lấy tốc độ thô và độ khớp chặt hơn với stack xung quanh.

## Khái niệm

### Agno

- Runtime Python, trước đây là Phi-data.
- "Không graph, không chain, không mẫu rối rắm — chỉ Python thuần."
- Mục tiêu hiệu năng theo tài liệu của họ: khởi tạo agent ~2μs, ~3.75 KiB bộ nhớ mỗi agent, ~23 nhà cung cấp model.
- Con đường production: backend FastAPI không trạng thái, phạm vi theo phiên. Mỗi request khởi tạo một agent mới; trạng thái phiên nằm trong DB.
- Đa phương thức gốc (text, image, audio, video, file) và agentic RAG.

Các mục tiêu tốc độ quan trọng khi bạn có hàng nghìn agent sống ngắn mỗi giây (fan-in chat, pipeline đánh giá). Chúng ít quan trọng hơn khi một agent chạy 10 phút.

### Mastra

- TypeScript, xây trên Vercel AI SDK.
- Ba primitive: **Agents**, **Tools** (có kiểu bằng Zod), **Workflows**.
- Unified Model Router — 3,300+ model trên 94 nhà cung cấp (tháng 3 năm 2026).
- Lưu trữ tổng hợp: memory, workflow, observability tới các backend khác nhau; ClickHouse được khuyến nghị cho observability ở quy mô lớn.
- Apache 2.0 với các thư mục `ee/` theo giấy phép doanh nghiệp source-available.
- Server adapter cho Express, Hono, Fastify, Koa; tích hợp Next.js và Astro hạng nhất.
- Có kèm Mastra Studio (localhost:4111) để debug.
- 22k+ star GitHub, 300k+ lượt tải npm mỗi tuần ở bản 1.0 (tháng 1 năm 2026).

### Định vị

Cả hai đều không cố trở thành LangGraph. Chúng cạnh tranh ở:

- **Độ khớp ngôn ngữ.** Agno cho nhóm ưu tiên Python; Mastra cho nhóm ưu tiên TypeScript.
- **Tính tiện dụng của runtime.** Agno = gần như không có chi phí phụ; Mastra = gắn với hệ sinh thái Vercel.
- **Observability.** Cả hai tích hợp với Langfuse/Phoenix/Opik (Bài 24) nhưng Mastra Studio là của chính chủ.

### Khi nào chọn cái nào

- **Agno** — backend Python, nhiều agent sống ngắn, yêu cầu hiệu năng cao, môi trường FastAPI.
- **Mastra** — backend TypeScript, triển khai Next.js / Vercel, định tuyến model đa nhà cung cấp hợp nhất, tool có kiểu bằng Zod.
- **LangGraph** (Bài 13) — khi trạng thái bền vững và suy luận graph tường minh quan trọng hơn tốc độ thô.
- **OpenAI / Claude Agent SDK** — khi bạn muốn hình dạng đã được sản phẩm hoá của nhà cung cấp (Bài 16–17).

### Mẫu hình này sai ở đâu

- **Tối ưu hiệu năng để mà tối ưu.** Chọn Agno vì "2μs" nghe hay trong khi khối lượng việc chỉ là một lời gọi agent chậm mỗi request. Chi phí phụ không phải nút thắt.
- **Khoá chặt vào hệ sinh thái.** Tích hợp mang hương Vercel của Mastra là điểm cộng trên Vercel, điểm trừ ở nơi khác.
- **Nhầm lẫn về giấy phép doanh nghiệp.** Các thư mục `ee/` của Mastra là source-available, không phải Apache 2.0. Hãy đọc giấy phép nếu bạn định fork.

```figure
wb-runtime-spawn
```

## Tự tay làm

Bài này chủ yếu mang tính so sánh — không một artifact code nào làm công bằng cho cả hai framework. Xem `code/main.py` để thấy một ví dụ đồ chơi đặt cạnh nhau: luồng tối giản "chạy agent, stream output, lưu phiên" được hiện thực hai lần (một theo kiểu Agno, một theo kiểu Mastra).

Chạy thử:

```
python3 code/main.py
```

Hai trace khác nhau về cấu trúc nhưng tương đương về chức năng.

## Áp dụng

- **Agno** — backend Python cần tốc độ và hình dạng FastAPI.
- **Mastra** — backend TypeScript với nhiều nhà cung cấp và primitive workflow.
- Cả hai đều có hook observability của chính chủ. Cả hai tích hợp với Langfuse.

## Đóng gói

`outputs/skill-runtime-picker.md` chọn Agno, Mastra, LangGraph, hoặc SDK của nhà cung cấp dựa trên stack, ngân sách độ trễ và hình dạng vận hành.

## Bài tập

1. Đọc tài liệu Agno. Chuyển vòng lặp ReAct dùng stdlib (Bài 01) sang Agno. Cái gì biến mất? Cái gì còn lại?
2. Đọc tài liệu Mastra. Chuyển vòng lặp tương tự sang Mastra. Kiểu của tool thay đổi ra sao (Zod so với không có gì)?
3. Đo hiệu năng: đo độ trễ khởi tạo agent trên stack của bạn. Con số 2μs của Agno có ý nghĩa với khối lượng việc của bạn không?
4. Thiết kế một cuộc di trú: nếu bạn đang chạy CrewAI bằng Python, điều gì sẽ vỡ khi chuyển sang Agno?
5. Đọc các điều khoản giấy phép `ee/` của Mastra. Những hạn chế nào sẽ ảnh hưởng tới một bản fork mã nguồn mở?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Agno | "Agent Python nhanh" | Runtime agent không trạng thái, phạm vi theo phiên |
| Mastra | "Agent TypeScript trên Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Truy cập đa nhà cung cấp" | Một client duy nhất cho 3,300+ model trên 94 nhà cung cấp |
| Composite storage | "Nhiều backend" | Memory/workflow/observability mỗi thứ tới một store khác nhau |
| Mastra Studio | "Trình debug cục bộ" | UI ở localhost:4111 để soi agent |
| Source-available | "Không phải OSS" | Giấy phép cho phép đọc mã nguồn nhưng hạn chế dùng thương mại |

## Đọc thêm

- [Tài liệu Agno Agent Framework](https://www.agno.com/agent-framework) — mục tiêu hiệu năng, tích hợp FastAPI
- [Tài liệu Mastra](https://mastra.ai/docs) — primitive, server adapter, Model Router
- [Tổng quan LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — lựa chọn thay thế graph có trạng thái
- [Comet Opik](https://www.comet.com/site/products/opik/) — các so sánh observability được tích hợp Mastra trích dẫn
