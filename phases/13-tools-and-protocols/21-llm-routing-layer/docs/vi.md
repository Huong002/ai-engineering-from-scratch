# Lớp routing (định tuyến) LLM — LiteLLM, OpenRouter, Portkey

> Khoá chặt vào một nhà cung cấp là đắt đỏ. Những khối lượng công việc gọi công cụ khác nhau phù hợp với những mô hình khác nhau. Gateway định tuyến cho bạn một bề mặt API duy nhất, retry, failover, theo dõi chi phí và guardrails. Ba kiểu mẫu thống trị năm 2026: LiteLLM (mã nguồn mở, tự vận hành), OpenRouter (SaaS được quản lý), Portkey (cấp production, mã nguồn mở từ tháng 3 năm 2026). Bài này nêu các tiêu chí ra quyết định và đi qua một gateway định tuyến bằng thư viện chuẩn.

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Giai đoạn 13, Bài 02 (function calling), Giai đoạn 13, Bài 17 (gateway)
**Time:** ~45 phút

## Mục tiêu học tập

- Phân biệt các lựa chọn định tuyến tự vận hành, được quản lý và cấp production.
- Cài đặt một chuỗi fallback (dự phòng) tự thử lại khi nhà cung cấp lỗi theo thứ tự ưu tiên xác định.
- Theo dõi chi phí và lượng token dùng cho từng request trên nhiều nhà cung cấp.
- Chọn giữa LiteLLM, OpenRouter và Portkey theo một ràng buộc production cụ thể.

## Vấn đề

Những tình huống mà định tuyến nhà cung cấp trở nên quan trọng:

1. **Chi phí.** Claude Sonnet có giá gấp 3 lần Haiku. Với một tác vụ phân loại, Haiku là đủ; với một tác vụ tổng hợp, Sonnet mới đáng tiền. Hãy định tuyến theo từng request.

2. **Failover.** OpenAI có một giờ không ổn. Mọi request đều thất bại. Bạn muốn tự động chuyển sang Anthropic mà không phải deploy lại.

3. **Độ trễ.** Một giao diện chat trực tiếp cần time-to-first-token nhanh. Một trình tóm tắt hàng loạt thì không. Định tuyến theo SLA độ trễ.

4. **Tuân thủ.** Người dùng EU phải ở lại vùng EU. Định tuyến theo vùng.

5. **Thử nghiệm.** A/B hai mô hình trên cùng khối lượng công việc. Định tuyến theo nhóm thử nghiệm.

Viết tay tất cả những thứ này cho từng tích hợp là lặp đi lặp lại. Một gateway định tuyến cho bạn một API tương thích OpenAI và lo phần còn lại.

## Khái niệm

### Hình dạng proxy tương thích OpenAI

Ai cũng nói tiếng OpenAI. Gateway định tuyến mở `/v1/chat/completions`, nhận schema của OpenAI, và bên trong chuyển tiếp tới Anthropic / Gemini / Cohere / Ollama / bất cứ thứ gì. Client không cần quan tâm.

### Bí danh mô hình

Thay vì một id snapshot cố định, code của bạn gọi `our_smart_model`. Gateway ánh xạ bí danh sang mô hình thật. Khi nhà cung cấp ra một thế hệ mới, bạn đổi bí danh ở phía server; code của bạn không phải sửa một dòng nào.

### Chuỗi fallback

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Gateway định nghĩa việc này trong một file cấu hình. Số lần retry được tính vào một ngân sách để các tầng fallback không làm chi phí bùng nổ.

### Cache ngữ nghĩa

Những prompt trùng khớp hoặc gần như trùng khớp sẽ trúng cache thay vì gọi nhà cung cấp. Mức tiết kiệm trên các vòng lặp agent lặp lại có thể là 30 đến 60 phần trăm. Khoá dựa trên embedding; các prompt gần như trùng nhau dùng chung một ô cache.

### Guardrails (rào chắn)

Ở cấp gateway:

- **Che PII.** Một lượt regex hoặc dựa trên ML trước khi gửi prompt.
- **Vi phạm chính sách.** Từ chối những prompt có nội dung bị cấm.
- **Bộ lọc đầu ra.** Rà soát kết quả trả về để tìm rò rỉ.

Portkey và Kong đều cung cấp guardrails theo quan điểm riêng của họ. LiteLLM để chúng thành tuỳ chọn.

### Giới hạn tốc độ theo từng khoá

Một API key = một team. Ngân sách theo từng khoá ngăn một team tiêu hết hạn mức chung. Hầu hết gateway đều hỗ trợ việc này.

### Đánh đổi giữa tự vận hành và được quản lý

| Yếu tố | LiteLLM (tự vận hành) | OpenRouter (được quản lý) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| Code | Mã nguồn mở, Python | SaaS được quản lý | Mã nguồn mở (tháng 3 năm 2026) + được quản lý |
| Cài đặt | Triển khai một proxy | Đăng ký | Cả hai đều được |
| Nhà cung cấp | 100+ | 300+ | 100+ |
| Thanh toán | Khoá của bạn | Tín dụng OpenRouter | Khoá của bạn |
| Khả năng quan sát | OpenTelemetry | Dashboard | OTel đầy đủ + che PII |
| Phù hợp nhất với | Team muốn toàn quyền kiểm soát | Prototype nhanh | Production có tuân thủ |

LiteLLM thắng khi bạn có một team SRE và muốn chủ quyền dữ liệu. OpenRouter thắng khi bạn muốn một gói đăng ký duy nhất và không phải lo hạ tầng. Portkey thắng khi bạn cần guardrails và tuân thủ có sẵn.

### Theo dõi chi phí

Mỗi request mang theo `provider`, `model`, `input_tokens`, `output_tokens`. Nhân với giá trên mỗi token của từng mô hình (lấy từ bảng giá mà gateway duy trì). Tổng hợp theo người dùng / team / dự án.

### MCP cộng với định tuyến

Một gateway có thể định tuyến cả lời gọi LLM VÀ các yêu cầu sampling của MCP. Khi modelPreferences của một yêu cầu sampling ưu tiên một mô hình cụ thể, gateway dịch sang backend phù hợp. Đây là chỗ Giai đoạn 13 · 17 (gateway MCP) và gateway định tuyến của bài này đôi khi hợp thành một dịch vụ duy nhất.

### Các chiến lược định tuyến

- **Ưu tiên tĩnh.** Mô hình đầu trong danh sách; chuyển sang cái sau khi lỗi.
- **Cân bằng tải.** Round-robin hoặc theo trọng số.
- **Theo chi phí.** Chọn mô hình rẻ nhất mà vẫn đạt độ trễ / chất lượng.
- **Theo độ trễ.** Chọn mô hình nhanh nhất trong N phút gần nhất.
- **Theo tác vụ.** Một bộ phân loại prompt đưa tác vụ viết code sang một mô hình, tóm tắt sang mô hình khác.

```figure
tp-router-failover
```

## Áp dụng

`code/main.py` cài đặt một gateway định tuyến trong ~150 dòng: nhận request theo hình dạng OpenAI, dịch sang các stub của từng nhà cung cấp, chạy một chuỗi fallback theo ưu tiên, theo dõi chi phí từng request, và áp một lượt che PII lên đầu vào. Chạy nó với ba kịch bản: request bình thường, nhà cung cấp chính gặp sự cố kích hoạt fallback, rò rỉ PII bị bắt bởi bước che.

Đáng chú ý:

- Dict `ROUTES`: bí danh -> danh sách nhà cung cấp cụ thể theo thứ tự ưu tiên.
- Vòng lặp fallback thử lại khi gặp 5xx.
- Bộ theo dõi chi phí nhân lượng token dùng với đơn giá từng mô hình.
- Bộ che PII rà và xoá các mẫu giống số SSN trước khi chuyển tiếp.

## Đóng gói

Bài này tạo ra `outputs/skill-routing-config-designer.md`. Cho một hồ sơ khối lượng công việc (độ trễ, chi phí, tuân thủ), skill chọn LiteLLM / OpenRouter / Portkey và tạo ra một cấu hình định tuyến.

## Bài tập

1. Chạy `code/main.py`. Kích hoạt kịch bản sự cố; xác nhận fallback rơi vào nhà cung cấp thứ hai và chi phí được quy đúng.

2. Thêm cache ngữ nghĩa: SHA256 của prompt là khoá tra cứu; cache hit trả về ngay lập tức. Đo mức tiết kiệm chi phí trên một lời gọi lặp lại.

3. Thêm một bộ phân loại prompt định tuyến các prompt "code ..." sang bí danh thiên về trí tuệ và các prompt "summarize ..." sang bí danh thiên về tốc độ.

4. Thiết kế ngân sách theo từng team: mỗi team có một hạn mức chi tiêu hàng tháng; gateway từ chối request khi chạm hạn mức. Chọn mức độ thực thi (theo từng request hay theo cửa sổ thời gian).

5. Đọc tài liệu của LiteLLM, OpenRouter và Portkey song song. Kể tên một tính năng mà mỗi bên có mà hai bên kia không có.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Routing gateway | "LLM proxy" | Lớp một bề mặt API đứng trước nhiều nhà cung cấp |
| OpenAI-compatible | "Speaks the OpenAI schema" | Nhận hình dạng `/v1/chat/completions`, dịch sang bất kỳ backend nào |
| Model alias | "our_smart_model" | Tên trong code của bạn mà gateway ánh xạ tới một mô hình cụ thể |
| Fallback chain | "Retry list" | Danh sách nhà cung cấp có thứ tự được thử khi lỗi |
| Semantic caching | "Prompt-embedding cache" | Khoá là embedding của prompt; các prompt gần trùng dùng chung một lần cache hit |
| Guardrails | "Input/output filters" | Che PII, từ chối vi phạm chính sách |
| Per-key rate limit | "Team budget" | Hạn mức gắn với một API key |
| Cost tracking | "Per-request spend" | Tổng hợp lượng token dùng x giá mỗi mô hình |
| LiteLLM | "The open proxy" | Gateway định tuyến mã nguồn mở, tự vận hành được |
| OpenRouter | "The managed SaaS" | Gateway được host với thanh toán dựa trên tín dụng |
| Portkey | "The production option" | Mã nguồn mở + được quản lý, có guardrails sẵn |

## Đọc thêm

- [LiteLLM — tài liệu](https://docs.litellm.ai/) — gateway định tuyến tự vận hành
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — SaaS định tuyến được quản lý
- [Portkey — tài liệu](https://portkey.ai/docs) — định tuyến cấp production với guardrails
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — hướng dẫn ra quyết định
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — khảo sát nhà cung cấp

---

[← Bài trước: OpenTelemetry GenAI](../../20-opentelemetry-genai/docs/vi.md) · [Bài sau: Agent Skills: hợp đồng portable và ranh giới thực thi →](../../22-skills-and-agent-sdks/docs/vi.md)
