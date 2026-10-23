# Toàn cảnh agent lập trình tự chủ (2026)

> SWE-bench Verified tăng từ 4% lên 80.9% trong chưa đầy ba năm. Cùng Claude Sonnet 4.5 đạt 43.2% với SWE-agent v1 và 59.8% với Cline autonomous — scaffolding (khung hỗ trợ) quanh mô hình nay quan trọng ngang chính mô hình. OpenHands (trước đây là OpenDevin) là nền tảng giấy phép MIT hoạt động sôi nổi nhất; vòng lặp CodeAct của nó chạy trực tiếp các hành động Python trong sandbox thay vì dùng lệnh gọi công cụ JSON. Các điểm số nổi bật che khuất một vấn đề phương pháp: 161 trong 500 tác vụ SWE-bench Verified chỉ cần sửa 1–2 dòng, còn SWE-bench Pro (tác vụ sửa 10+ dòng) chỉ đạt 23–59% với cùng các mô hình hàng đầu.

**Type:** Learn
**Languages:** Python (stdlib, CodeAct vs JSON tool-call comparison)
**Prerequisites:** Giai đoạn 14, Bài 07 (Sử dụng công cụ), Giai đoạn 15, Bài 01 (Agent dài hạn)
**Time:** ~45 phút

## Vấn đề

“Agent lập trình nào tốt nhất” là câu hỏi sai. Câu hỏi đúng là: với tập tác vụ giống công việc của tôi và scaffolding tôi sẽ chạy trong môi trường thực tế, độ tin cậy từ đầu đến cuối là bao nhiêu?

Từ 2022 đến 2026, giới nghiên cứu nhận ra scaffolding — lớp truy xuất, bộ lập kế hoạch, sandbox, vòng lặp sửa-kiểm tra, định dạng phản hồi — là yếu tố cốt yếu. Claude Sonnet 4.5 trên SWE-agent v1 đạt 43.2% ở SWE-bench Verified; cùng mô hình trong scaffolding tự chủ của Cline đạt 59.8%. Chênh lệch 16.6 điểm tuyệt đối, cùng bộ trọng số. Mô hình nền chỉ là một thành phần; vòng lặp mới là sản phẩm.

Vấn đề đi kèm là benchmark gần bão hòa che giấu sự thụt lùi. SWE-bench Verified đã gần bão hòa, và nhóm tác vụ dễ (161 trong 500 tác vụ cần sửa ≤2 dòng) đẩy điểm cao nhất lên. Chất lượng thực tế được đo tốt hơn trên các tập như SWE-bench Pro (sửa 10+ dòng), nơi cùng các hệ thống dẫn đầu vẫn chỉ đạt 23–59%.

## Khái niệm

### SWE-bench trong một đoạn

SWE-bench (Jimenez et al.) lấy các issue GitHub thực có bản vá chuẩn và yêu cầu agent tạo bản vá giúp bộ kiểm thử chạy thành công. SWE-bench Verified (OpenAI, 2024) là tập con 500 tác vụ được con người tuyển chọn, đã loại tác vụ mơ hồ và bị lỗi. SWE-bench Pro là phiên bản kế tiếp khó hơn — tác vụ cần sửa 10+ dòng, nơi các agent hàng đầu hiện đạt 23–59%.

### Đường tăng từ 2022 → 2026 thực sự cho thấy gì

- **2022**: mô hình nghiên cứu đạt ~4% trên SWE-bench gốc.
- **2024**: GPT-4 + scaffolding kiểu Devin đạt ~14%; SWE-agent đạt ~12%.
- **2025**: Claude 3.5/3.7 Sonnet trong Aider và SWE-agent nâng kết quả lên mức 40–55%.
- **2026**: Claude Sonnet 4.5 và các đối thủ hàng đầu đạt 70–80%+ trên SWE-bench Verified. Bảng xếp hạng của Epoch AI theo dõi kết quả này trực tiếp.

Đà tăng đến từ ba nguồn cộng hưởng: mô hình nền tốt hơn, scaffolding tốt hơn (CodeAct, tự phản hồi, vòng lặp kiểm tra), và benchmark tốt hơn (Verified loại bỏ nhiễu).

### CodeAct so với lệnh gọi công cụ JSON

OpenHands (All-Hands-AI, arXiv:2407.16741, trước đây là OpenDevin) chọn một kiến trúc cụ thể: thay vì để mô hình phát lệnh gọi công cụ JSON cho hệ thống chủ giải mã và chạy, mô hình phát mã Python để một kernel kiểu Jupyter chạy trong sandbox. Agent có thể lặp qua các tệp, nối các công cụ và tự bắt lỗi trong một hành động.

Đánh đổi là:

- **Lệnh gọi công cụ JSON**: mỗi hành động tốn một lượt; dễ kiểm tra; khó kết hợp nhiều thao tác; mặc định an toàn vì mỗi lệnh gọi đều qua bộ xác thực tường minh.
- **CodeAct**: một hành động có thể là cả chương trình; dễ kết hợp thao tác; cần sandbox được bảo vệ nghiêm ngặt (OpenHands dùng Docker để cách ly); kiểu lỗi có thể là bất cứ điều gì môi trường chạy sandbox cho phép.

Cả hai kiến trúc đều được dùng thực tế. CodeAct phổ biến nhất ở các nền tảng mở (OpenHands, smolagents). Lệnh gọi công cụ JSON vẫn phổ biến nhất ở dịch vụ được quản lý (Anthropic Managed Agents, OpenAI Assistants), nơi nhà cung cấp kiểm soát bộ thực thi.

### Các scaffolding trong bức tranh 2026

| Scaffolding | Giấy phép | Cách thực thi | Đặc điểm nổi bật |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | CodeAct trong Docker | Nền tảng mở hoạt động sôi nổi nhất; có thể phát lại luồng sự kiện |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | Scaffolding SWE-bench đầu tiên chạy từ đầu đến cuối |
| Aider | Apache-2 | sửa bằng diff trong repo cục bộ | Scaffolding gọn, ít bị thụt lùi hiệu năng |
| Cline | Apache-2 | Agent VS Code với chính sách công cụ | Scaffolding mở đạt điểm cao nhất trên Sonnet 4.5 |
| Devin (Cognition) | Proprietary | VM được quản lý + bộ lập kế hoạch | Dòng sản phẩm “kỹ sư phần mềm AI” đầu tiên |
| Claude Code | Proprietary | Chế độ cấp quyền + quy trình | Bài 10 giải thích chi tiết vòng lặp agent |

### Vì sao scaffolding chi phối kết quả

Một lượt chạy lập trình là chuỗi hành động dài hạn (Bài 1). Độ tin cậy chịu tác động cộng dồn qua từng bước. Scaffolding cải thiện điểm ở ba chỗ:

1. **Truy xuất**: tìm đúng tệp để đọc là nút thắt khó thấy. ACI của SWE-agent, chỉ mục tệp của OpenHands và bản đồ repo của Aider đều giải quyết việc này.
2. **Vòng lặp kiểm tra**: chạy kiểm thử, đọc stack trace rồi thử lại tạo chênh lệch 10+ điểm trên SWE-bench.
3. **Khoanh vùng lỗi**: sandbox có thể hoàn tác khi lỗi giúp tránh thiệt hại cộng dồn. Cùng mô hình, có và không có vòng lặp kiểm tra, trông như hai sản phẩm khác nhau.

### Benchmark bão hòa và tập tác vụ thực tế

Các tác giả OpenHands và Epoch AI đều lưu ý SWE-bench Verified có nhóm tác vụ dễ: 161 trong 500 tác vụ chỉ cần sửa 1–2 dòng. Điểm cao một phần nhờ nhóm này. SWE-bench Pro chỉ gồm tác vụ sửa 10+ dòng và ngay cả hệ thống hàng đầu cũng chỉ đạt 23–59%. Tập tác vụ thực tế của bạn gần như chắc chắn giống Pro hơn Verified.

Hệ quả khi chọn agent: hãy chạy một tập con giống Pro từ các lỗi tồn đọng của chính bạn. Điểm quan trọng là điểm trên những tác vụ đại diện cho sản phẩm bạn bàn giao.

```figure
a5-scaffold-delta
```

## Áp dụng

`code/main.py` so sánh hai scaffolding agent đơn giản trên một tập tác vụ nhỏ cố định:

1. Scaffolding **lệnh gọi công cụ JSON** thực hiện một hành động mỗi lượt.
2. Scaffolding **CodeAct** có thể phát một đoạn Python ngắn mỗi hành động.

Cả hai dùng một “mô hình” giả (quy tắc cố định) để phép so sánh tách tác động của scaffolding khỏi chất lượng mô hình. Kết quả cho thấy scaffolding CodeAct giải được nhiều tác vụ hơn trong ít lượt hơn, nhưng mỗi hành động có phạm vi tác động lớn hơn.

## Đóng gói

`outputs/skill-scaffold-audit.md` giúp bạn đánh giá một scaffolding agent lập trình trước khi áp dụng: chất lượng truy xuất, có bộ kiểm tra hay không, cách ly sandbox và mức phù hợp giữa benchmark với tập tác vụ thực tế.

## Bài tập

1. Chạy `code/main.py`. Mỗi scaffolding cần bao nhiêu lượt trên cùng tập tác vụ? Phạm vi tác động của mỗi hành động ở từng loại là bao nhiêu?

2. Đọc bài báo OpenHands (arXiv:2407.16741). Bài báo lập luận CodeAct vượt lệnh gọi công cụ JSON ở tác vụ phức tạp. Nêu một kiểu lỗi bài báo thừa nhận và viết một câu về lúc kiểu lỗi đó sẽ chi phối trong môi trường thực tế.

3. Chọn một tác vụ trong danh sách lỗi tồn đọng cần sửa 10+ dòng trên hai tệp. Ước tính xác suất thành công từ đầu đến cuối của một mô hình hàng đầu khi dùng (a) lệnh gọi công cụ JSON và (b) CodeAct. Giải thích chênh lệch.

4. SWE-bench Verified có 161 tác vụ một tệp, sửa 1–2 dòng. Hãy tính điểm sau khi loại chúng. Thứ hạng thay đổi thế nào?

5. Đọc "Introducing SWE-bench Verified" (OpenAI). Giải thích phương pháp cụ thể dùng để loại tác vụ mơ hồ và nêu một loại tác vụ mà quá trình tuyển chọn sẽ bỏ sót.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| SWE-bench | “Benchmark lập trình” | Các issue GitHub thực có bản vá chuẩn và bộ kiểm thử |
| SWE-bench Verified | “Tập con đã làm sạch” | 500 tác vụ do con người tuyển chọn, vẫn có nhóm tác vụ dễ |
| SWE-bench Pro | “Tập con khó hơn” | Sửa 10+ dòng; hệ thống hàng đầu đạt 23–59% |
| CodeAct | “Mã là hành động” | Agent phát Python; kernel kiểu Jupyter chạy trong sandbox |
| JSON tool call | “Gọi hàm” | Mỗi hành động là một payload JSON có cấu trúc, được xác thực trước khi chạy |
| Scaffold | “Khung agent” | Lớp truy xuất + bộ lập kế hoạch + bộ thực thi + vòng lặp kiểm tra quanh mô hình nền |
| ACI (Agent-Computer Interface) | “Định dạng của SWE-agent” | Bộ lệnh thiết kế để LLM dễ dùng, không phải cho shell của con người |
| Verifier loop | “Kiểm thử rồi thử lại” | Chạy kiểm thử, đọc kết quả, sửa bản vá; cải thiện độ tin cậy lớn nhất ngoài mô hình |

## Đọc thêm

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — benchmark gốc và phương pháp.
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — cách xây dựng tập con được tuyển chọn.
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — kiến trúc CodeAct và thiết kế luồng sự kiện.
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — điểm số được theo dõi trực tiếp.
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — cách nhìn về độ tin cậy của agent lập trình dài hạn.

---

[← Bài trước: Thiết kế tự cải thiện có giới hạn](../../08-bounded-self-improvement/docs/vi.md) · [Bài sau: Chế độ cấp quyền cho agent tự chủ →](../../10-claude-code-permission-modes/docs/vi.md)
