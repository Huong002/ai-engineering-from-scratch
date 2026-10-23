# Prompt caching và cache ngữ cảnh

> System prompt của bạn là 4.000 token. Ngữ cảnh RAG của bạn là 20.000 token. Bạn gửi cả hai trong mọi request. Và bạn cũng trả tiền cho cả hai — mỗi lần. Prompt caching (cache prompt) cho phép nhà cung cấp giữ ấm prefix (tiền tố) đó ở phía họ và chỉ tính bạn 10% mức giá thường khi dùng lại. Dùng đúng cách, nó cắt chi phí suy luận 50–90% và độ trễ token đầu tiên 40–85%.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 11, Bài 01 (Kỹ thuật viết prompt), Giai đoạn 11, Bài 05 (Kỹ nghệ ngữ cảnh), Giai đoạn 11, Bài 11 (Cache và chi phí)
**Time:** ~60 phút

## Vấn đề

Một agent lập trình gửi cùng một system prompt 15.000 token tới Claude ở mọi lượt của cuộc hội thoại. Hai mươi lượt với giá $3/M token đầu vào là $0,90 chỉ riêng chi phí đầu vào — chưa tính bất kỳ tin nhắn thật nào của người dùng. Nhân với 10.000 cuộc hội thoại mỗi ngày và hoá đơn chạm $9.000/ngày cho phần văn bản không bao giờ thay đổi.

Bạn không thể thu nhỏ prompt mà không làm giảm chất lượng. Bạn không thể tránh việc gửi nó — mô hình cần nó ở mọi lượt. Cách duy nhất là ngừng trả đủ giá cho một tiền tố mà nhà cung cấp đã thấy rồi.

Đó chính là prompt caching. Anthropic phát hành nó tháng 8 năm 2024 (với biến thể TTL (thời gian sống) mở rộng 1 giờ vào 2025), OpenAI tự động hoá nó cuối năm đó, Google phát hành context caching tường minh song song với Gemini 1.5, và cả ba giờ đều cung cấp nó như một tính năng hạng nhất trên các mô hình tiên phong của họ.

## Khái niệm

![Cache prompt: ghi một lần, đọc rẻ](../assets/prompt-caching.svg)

**Cơ chế.** Khi prefix của một request khớp với prefix của một request gần đây, nhà cung cấp phục vụ KV cache từ lần chạy trước thay vì mã hoá lại các token. Bạn trả một khoản phụ phí ghi nhỏ ở lần đầu và được giảm giá đọc lớn ở mọi lần sau.

**Ba kiểu nhà cung cấp vào năm 2026.**

| Nhà cung cấp | Kiểu API | Giảm giá khi trúng | Phụ phí ghi | TTL mặc định | Tối thiểu để cache |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | Marker `cache_control` tường minh trên các content block | giảm 90% đầu vào | phụ phí 25% | 5 phút (mở rộng được tới 1 giờ) | 1,024 tokens (Sonnet/Opus), 2,048 (Haiku) |
| OpenAI | Tự động phát hiện prefix | giảm 50% đầu vào | không | Tối đa 1 giờ (nỗ lực tốt nhất) | 1,024 tokens |
| Google (Gemini) | API `CachedContent` tường minh | Tính phí lưu trữ; đọc ở ~25% mức thường | Phí lưu trữ mỗi token·giờ | Người dùng đặt (mặc định 1 giờ) | 4,096 tokens (Flash), 32,768 (Pro) |

**Bất biến.** Cả ba đều chỉ cache tiền tố. Nếu bất kỳ token nào khác nhau giữa các request, mọi thứ sau token khác biệt đầu tiên đều là cache miss (trượt cache). Đặt phần *ổn định* ở trên, phần *thay đổi* ở dưới.

### Bố cục thân thiện với cache

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

Phá vỡ thứ tự — đặt tin nhắn người dùng lên trên system prompt, xen các truy xuất động vào giữa các few-shot — và cache sẽ không bao giờ trúng (cache hit).

### Cách tính điểm hoà vốn

Phụ phí ghi 25% của Anthropic nghĩa là một block được cache phải được đọc ít nhất hai lần mới thực sự tiết kiệm chi phí (cost saving). 1 lần ghi + 1 lần đọc trung bình tốn 0.675x chi phí mỗi request (tiết kiệm 32%); 1 lần ghi + 10 lần đọc trung bình 0.205x (tiết kiệm 80%). Quy tắc ngón tay cái: hãy cache mọi thứ bạn dự kiến dùng lại ít nhất 3 lần trong khoảng TTL.

```figure
prompt-cache-hit
```

## Tự tay làm

### Bước 1: Prompt caching của Anthropic với marker tường minh

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

Marker `cache_control` bảo Anthropic lưu block trong 5 phút. Dùng lại trong khoảng đó thì trúng cache; dùng lại sau khi hết hạn thì phải ghi lại.

**Các trường usage trong response:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

Kiểm tra cả hai trường trong CI — nếu `cache_read_input_tokens` vẫn bằng 0 qua các request, khoá cache của bạn đang bị trôi.

### Bước 2: TTL mở rộng một giờ

Với các batch job chạy lâu, mặc định 5 phút sẽ hết hạn giữa các job. Đặt `ttl`:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

TTL 1 giờ tốn gấp 2 lần phụ phí ghi (50% so với mức cơ sở thay vì 25%) nhưng hoàn vốn nhanh với bất kỳ batch nào dùng lại tiền tố hơn 5 lần.

### Bước 3: Cache tự động của OpenAI

OpenAI không cho bạn gì để cấu hình. Mọi prefix dài hơn 1.024 token mà khớp với một request gần đây sẽ tự động được giảm 50%.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

Quy tắc bố cục thân thiện với cache vẫn áp dụng. Hai thứ giết cache của OpenAI mà không giết cache của Anthropic: thay đổi trường `user` (được dùng như một thành phần của khoá cache) và sắp xếp lại tools.

### Bước 4: Context caching tường minh của Gemini

Gemini coi cache là một đối tượng hạng nhất mà bạn tạo ra và đặt tên:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3.8-flash",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini tính phí lưu trữ theo token·giờ chừng nào cache còn sống, và đọc ở ~25% mức giá đầu vào thường. Đây là hình thức phù hợp khi bạn dùng lại cùng một prompt khổng lồ qua nhiều phiên trong nhiều ngày.

### Bước 5: đo tỉ lệ trúng trong môi trường thực

Xem `code/main.py` để thấy một bộ tính toán mô phỏng cho ba nhà cung cấp, theo dõi số lần ghi/đọc/trượt và tính chi phí pha trộn trên mỗi 1K request. Hãy chặn deploy dựa trên tỉ lệ trúng mục tiêu — hầu hết các thiết lập Anthropic trong môi trường thực nên thấy phần đọc >80% sau khi đã khởi động ấm.

## Những cạm bẫy vẫn còn trong năm 2026

- **Timestamp động ở trên cùng.** `"Current time: 2026-04-22 15:30:02"` ở đầu system prompt. Mỗi request đều trượt vì cache bị vô hiệu hoá (invalidation) ngay từ token đầu. Hãy chuyển timestamp xuống dưới điểm ngắt cache.
- **Sắp xếp lại tool.** Hãy serialize tools theo một thứ tự ổn định — một lần xáo trộn dict giữa các lần deploy sẽ phá mọi lần trúng cache.
- **Văn bản gần trùng.** "You are helpful." so với "You are a helpful assistant." — khác một byte = trượt toàn bộ.
- **Block quá nhỏ.** Anthropic áp mức sàn 1.024 token (2.048 với Haiku). Block nhỏ hơn sẽ âm thầm không được cache.
- **Dashboard chi phí mù.** Tách "input tokens" thành đã cache và chưa cache. Nếu không, lưu lượng giảm sẽ trông như một chiến thắng của cache.

## Áp dụng

Bộ công cụ cache năm 2026:

| Tình huống | Chọn |
|-----------|------|
| Agent với system prompt 10k+ ổn định, nhiều lượt | `cache_control` của Anthropic với TTL 5 phút |
| Batch job dùng lại một tiền tố trong 30+ phút | Anthropic với `ttl: "1h"` |
| Endpoint serverless trên GPT-5, không có hạ tầng tuỳ chỉnh | OpenAI tự động (chỉ cần giữ prefix ổn định và dài) |
| Dùng lại nhiều ngày một kho mã/tài liệu khổng lồ | `CachedContent` tường minh của Gemini |
| Fallback giữa các nhà cung cấp | Giữ bố cục tiền tố có thể cache giống nhau ở mọi nhà cung cấp để lần trúng nào cũng dùng được |

Kết hợp với semantic caching (Giai đoạn 11 · 11) cho lớp tin nhắn người dùng: prompt caching xử lý việc dùng lại *trùng token*, semantic caching xử lý việc dùng lại *trùng nghĩa*.

## Đóng gói

Lưu `outputs/skill-prompt-caching-planner.md`:

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## Bài tập

1. **Dễ.** Lấy một cuộc hội thoại 10 lượt với system prompt 5.000 token gọi Claude. Chạy không có `cache_control` rồi chạy có. Báo cáo hoá đơn token đầu vào cho mỗi trường hợp.
2. **Trung bình.** Viết một bộ khung test mà, khi cho trước một mẫu prompt và một log request, tính tỉ lệ trúng kỳ vọng và số tiền tiết kiệm theo từng nhà cung cấp (Anthropic 5m, Anthropic 1h, OpenAI tự động, Gemini tường minh).
3. **Khó.** Xây một bộ tối ưu bố cục: cho trước một prompt và danh sách các trường được đánh dấu `stable=True/False`, viết lại prompt để đặt một điểm ngắt cache duy nhất ở vị trí thân thiện với cache nhất mà không mất thông tin. Kiểm chứng trên một endpoint Anthropic thật.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Prompt caching | "Làm prompt dài thành rẻ" | Dùng lại KV cache phía nhà cung cấp cho các tiền tố khớp nhau; giảm 50-90% trên các token đầu vào lặp lại. |
| `cache_control` | "Marker của Anthropic" | Thuộc tính của content block khai báo "mọi thứ tới đây đều có thể cache"; `{"type": "ephemeral"}`. |
| Cache write | "Trả phụ phí" | Request đầu tiên làm đầy cache; tính giá ~1.25x mức đầu vào trên Anthropic, miễn phí trên OpenAI. |
| Cache read | "Phần giảm giá" | Các request sau khớp tiền tố; tính 10% (Anthropic), 50% (OpenAI), ~25% (Gemini). |
| TTL | "Nó sống bao lâu" | Số giây cache còn ấm; Anthropic mặc định 5m (mở rộng được 1h), OpenAI nỗ lực tốt nhất tới 1h, Gemini do người dùng đặt. |
| Extended TTL | "Cache 1 giờ của Anthropic" | `{"type": "ephemeral", "ttl": "1h"}`; gấp 2 phụ phí ghi nhưng đáng với việc dùng lại theo batch. |
| Prefix match | "Vì sao cache của tôi trượt" | Cache chỉ trúng khi mọi token từ đầu tới điểm ngắt đều giống hệt từng byte. |
| Context caching (Gemini) | "Cái tường minh" | Đối tượng cache có tên, tính phí lưu trữ của Google; tốt nhất cho việc dùng lại kho dữ liệu lớn trong nhiều ngày. |

## Đọc thêm

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`, TTL 1h, các bảng điểm hoà vốn.
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — khớp prefix tự động.
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — API `CachedContent` và giá lưu trữ.
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — bài công bố gốc kèm số liệu độ trễ.
- Giai đoạn 11 · 05 (Kỹ nghệ ngữ cảnh) — nên cắt prompt ở đâu để cache có thể ăn.
- Giai đoạn 11 · 11 (Cache và chi phí) — ghép prompt caching với một semantic cache trên tin nhắn người dùng.
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — mô hình bộ nhớ KV cache mà prompt caching phơi ra cho người dùng; giải thích vì sao đọc lại một tiền tố đã cache rẻ hơn ~10× so với tính lại.
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill là pha mà prompt caching đi tắt; bài này giải thích vì sao TTFT giảm mạnh khi trúng cache trong khi TPOT không đổi.
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — prompt caching nằm cạnh speculative decoding, Flash Attention và MQA/GQA như những đòn bẩy bẻ cong đường chi phí suy luận; đọc bài này để biết ba đòn bẩy còn lại.

---

[← Bài trước: Model Context Protocol (MCP)](../../14-model-context-protocol/docs/vi.md) · [Bài sau: Máy trạng thái (state machine) của agent →](../../16-langgraph-state-machines/docs/vi.md)
