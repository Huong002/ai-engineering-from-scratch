# Harness như một thư viện — Subagent (agent con) và Session Store

> Một harness bạn có thể import: công cụ dựng sẵn, subagent để cô lập ngữ cảnh, hook (móc vòng đời), lan truyền trace W3C, lưu trữ session (phiên). Claude Agent SDK là ví dụ tham chiếu — dạng thư viện của harness Claude Code — còn Claude Managed Agents là lựa chọn được host sẵn cho công việc async dài hạn.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 01 (vòng lặp agent), Giai đoạn 14, Bài 10 (thư viện kỹ năng)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích khác biệt giữa Anthropic Client SDK (API thô) và Claude Agent SDK (hình dạng harness).
- Mô tả subagent — song song hoá và cô lập ngữ cảnh — và khi nào nên dùng.
- Kể tên bề mặt session store của Python SDK (`append`, `load`, `list_sessions`, `delete`, `list_subkeys`) và vai trò của `--session-mirror`.
- Xây dựng một harness stdlib với công cụ dựng sẵn, spawn subagent với ngữ cảnh cô lập, hook vòng đời, và session store.

## Vấn đề

Một API LLM thô chỉ cho bạn một lượt gọi. Một agent production cần thực thi công cụ, MCP server, hook vòng đời, spawn subagent, lưu trữ session, lan truyền trace. Claude Agent SDK mang hình dạng đó dưới dạng thư viện — đúng harness mà Claude Code dùng, mở ra cho agent tuỳ chỉnh.

## Khái niệm

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`).** Messages API thô. Bạn tự lo vòng lặp, công cụ, trạng thái.
- **Agent SDK (`claude-agent-sdk`).** Thực thi công cụ dựng sẵn, kết nối MCP, hook, spawn subagent, session store. Vòng lặp Claude Code dưới dạng thư viện.

### Công cụ dựng sẵn

SDK mang sẵn hơn 10 công cụ: đọc/ghi file, shell, grep, glob, web fetch, v.v. Công cụ tuỳ chỉnh đăng ký qua giao diện tool-schema chuẩn.

### Subagent

Hai mục đích được Anthropic ghi rõ:

1. **Song song hoá.** Chạy công việc độc lập đồng thời. "Tìm file test cho từng module trong 20 module này" là 20 task subagent song song.
2. **Cô lập ngữ cảnh.** Subagent dùng cửa sổ ngữ cảnh riêng; chỉ kết quả trả về orchestrator. Ngân sách của orchestrator được giữ nguyên.

Python SDK mới bổ sung: `list_subagents()`, `get_subagent_messages()` để đọc bản ghi subagent.

### Session store

Tương thích giao thức với TypeScript:

- `append(session_id, message)` — thêm một lượt.
- `load(session_id)` — khôi phục hội thoại.
- `list_sessions()` — liệt kê.
- `delete(session_id)` — xoá theo cascade xuống session của subagent.
- `list_subkeys(session_id)` — liệt kê key của subagent.

`--session-mirror` (cờ CLI) sao bản ghi ra file ngoài ngay khi stream, để gỡ lỗi.

### Hook

Các hook vòng đời bạn có thể đăng ký:

- `PreToolUse`, `PostToolUse` — chặn hoặc kiểm tra các lệnh gọi công cụ.
- `SessionStart`, `SessionEnd` — thiết lập và dọn dẹp.
- `UserPromptSubmit` — xử lý input của người dùng trước khi model thấy.
- `PreCompact` — chạy trước khi nén ngữ cảnh.
- `Stop` — dọn dẹp khi agent thoát.
- `Notification` — cảnh báo qua kênh phụ.

Hook là cách pro-workflow (tham chiếu chương trình Giai đoạn 14) và các hệ thống tương tự thêm hành vi xuyên suốt.

### Ngữ cảnh trace W3C

Các span OTel đang hoạt động ở nơi gọi được lan truyền vào tiến trình con CLI qua header trace context W3C. Toàn bộ trace đa tiến trình hiện ra thành một trace duy nhất trên backend của bạn.

### Claude Managed Agents

Lựa chọn được host sẵn (header beta `managed-agents-2026-04-01`). Công việc async dài hạn, prompt caching dựng sẵn, nén ngữ cảnh dựng sẵn. Đánh đổi quyền kiểm soát để lấy hạ tầng được quản lý.

### Chỗ mẫu này hay sai

- **Spawn quá nhiều subagent.** Spawn 100 subagent cho 100 task tí xíu. Chi phí phụ trội lấn át. Hãy gom lô.
- **Hook phình ra.** Mỗi team thêm hook; thời gian khởi động tăng vọt. Rà lại hook mỗi quý.
- **Session phình ra.** Session tích tụ; kích thước tăng. Dùng `list_sessions` + chính sách hết hạn.

```figure
ae-subagent-isolation
```

## Tự tay làm

`code/main.py` cài đặt hình dạng SDK bằng stdlib:

- `Tool`, `ToolRegistry` với `read_file`, `write_file`, `list_dir` dựng sẵn.
- `Subagent` — ngữ cảnh riêng, chạy cô lập, trả kết quả về.
- `SessionStore` — append, load, list, delete, list_subkeys.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- Bản demo: agent chính spawn 3 subagent song song (mỗi cái cô lập), tổng hợp kết quả, lưu session.

Chạy thử:

```
python3 code/main.py
```

Trace cho thấy cô lập ngữ cảnh của subagent (kích thước ngữ cảnh orchestrator vẫn có chặn trên), việc chạy hook, và lưu session.

## Áp dụng

- **Claude Agent SDK** cho sản phẩm ưu tiên Claude muốn hình dạng harness Claude Code.
- **Claude Managed Agents** cho công việc async dài hạn được host sẵn.
- **OpenAI Agents SDK** (Bài 16) cho sản phẩm đối trọng ưu tiên OpenAI.
- **LangGraph + công cụ tuỳ chỉnh** nếu bạn muốn dạng máy trạng thái hình đồ thị.

## Đóng gói

`outputs/skill-claude-agent-scaffold.md` dựng khung một ứng dụng Claude Agent SDK với subagent, hook, session store, gắn MCP server, và lan truyền trace W3C.

## Bài tập

1. Thêm bộ spawn subagent gom 20 task thành nhóm 5 subagent song song. Đo kích thước ngữ cảnh orchestrator so với một-subagent-mỗi-task.
2. Cài hook `PreToolUse` giới hạn tần suất gọi `write_file` (5 lần mỗi phút mỗi session). Theo dõi hành vi.
3. Nối `list_subkeys` để vẽ cây subagent. Lồng sâu trông thế nào?
4. Chuyển bản đồ chơi sang gói Python `claude-agent-sdk` thật. Việc đăng ký công cụ thay đổi ra sao?
5. Đọc tài liệu Claude Managed Agents. Khi nào bạn nên chuyển từ tự host sang managed?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Agent SDK | "Claude Code như một thư viện" | Hình dạng harness: công cụ, MCP, hook, subagent, session store |
| Subagent | "Agent con" | Ngữ cảnh riêng, ngân sách riêng; kết quả trào lên trên |
| Session store | "DB hội thoại" | Lưu, nạp, liệt kê, xoá lượt với cascade xuống subagent |
| Hook | "Callback vòng đời" | Trước/sau công cụ, session, prompt submit, compact, stop |
| W3C trace context | "Trace xuyên tiến trình" | Span cha lan truyền vào tiến trình con CLI |
| Managed Agents | "Harness được host" | Công việc async dài hạn do Anthropic host |
| `--session-mirror` | "Bản sao bản ghi" | Ghi các lượt session ra file ngoài ngay khi stream |
| MCP server | "Bề mặt công cụ" | Nguồn công cụ/tài nguyên ngoài gắn vào agent |

## Đọc thêm

- [Tổng quan Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — dạng thư viện của Claude Code
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — các mẫu production
- [Tổng quan Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) — lựa chọn được host sẵn
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — đối trọng
