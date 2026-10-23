# Kiểu hỏng: Vì sao agent vỡ

> MASFT (Berkeley, 2025) liệt kê 14 failure mode (kiểu hỏng) đa tác tử trong 3 nhóm. Bộ phân loại của Microsoft ghi lại cách các lỗi AI sẵn có khuếch đại trong bối cảnh agent. Dữ liệu thực địa của ngành hội tụ về năm kiểu lặp lại: hành động bịa, lấn phạm vi, lỗi lan truyền, mất ngữ cảnh, dùng sai công cụ.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Giai đoạn 14, Bài 05 (Self-Refine và CRITIC), Giai đoạn 14, Bài 24 (khả năng quan sát)
**Time:** ~60 phút

## Mục tiêu học tập

- Kể tên ba nhóm kiểu hỏng của MASFT và ít nhất bốn kiểu hỏng cụ thể trong mỗi nhóm.
- Giải thích vì sao lỗi ở agent khuếch đại các kiểu hỏng AI sẵn có (thiên lệch, ảo giác).
- Mô tả năm kiểu hỏng lặp lại trong ngành và cách giảm thiểu chúng.
- Cài đặt một bộ phát hiện dùng stdlib, gán nhãn kiểu hỏng cho trace của agent.

## Vấn đề

Các đội đưa agent vào chạy ổn trên 90% trace. 10% hỏng không phải nhiễu ngẫu nhiên — chúng rơi vào một số ít nhóm lặp lại. Một khi gọi được tên, bạn có thể giám sát và sửa chúng.

## Khái niệm

### MASFT (Berkeley, arXiv:2503.13657)

Bộ phân loại kiểu hỏng của hệ đa tác tử (Multi-Agent System Failure Taxonomy). 14 failure mode chia thành 3 nhóm. Cohen's Kappa giữa những người gán nhãn là 0.88 — các nhóm phân biệt được một cách đáng tin cậy.

Luận điểm trung tâm: kiểu hỏng là lỗi thiết kế cốt lõi trong hệ đa tác tử, không phải hạn chế của LLM cần sửa bằng mô hình nền tốt hơn.

### Microsoft Taxonomy of Failure Mode in Agentic AI Systems

- Các lỗi AI sẵn có (thiên lệch, ảo giác, rò rỉ dữ liệu) khuếch đại trong bối cảnh agent.
- Lỗi mới nảy sinh từ quyền tự chủ: hành động ngoài ý muốn ở quy mô lớn, dùng sai công cụ, mission drift (lệch hướng nhiệm vụ).
- Whitepaper này chính là sổ đăng ký rủi ro cho sản phẩm agent.

### Characterizing Faults in Agentic AI (arXiv:2603.06847)

- Lỗi phát sinh từ điều phối, tiến hoá trạng thái bên trong và tương tác với môi trường.
- Không chỉ là "code dở" hay "đầu ra mô hình dở".

### LLM Agent Hallucinations Survey (arXiv:2509.18970)

Hai biểu hiện chính:

1. **Lệch khỏi chỉ dẫn (Instruction-following Deviation)** — agent không tuân theo system prompt.
2. **Dùng sai ngữ cảnh tầm xa (Long-range Contextual Misuse)** — agent quên hoặc áp dụng nhầm ngữ cảnh từ các lượt trước.

Lỗi ở cấp ý định con: Bỏ sót (bước bị bỏ), Dư thừa (bước lặp lại), Sai thứ tự (bước không đúng trình tự).

### Năm kiểu hỏng lặp lại trong ngành

Arize, Galileo, NimbleBrain với các phân tích thực địa 2024-2026 đều hội tụ về:

1. **Hallucinated tool call (lượt gọi công cụ bịa).** Agent gọi một công cụ không tồn tại hoặc bịa ra tham số.
2. **Scope creep.** Agent mở rộng nhiệm vụ vượt yêu cầu của người dùng (tạo thêm PR, gửi thêm email).
3. **Cascading errors.** Một lượt gọi sai kéo theo hiệu ứng hạ nguồn. Một SKU ảo do ảo giác sinh ra kích hoạt bốn lượt gọi API — thành sự cố đa hệ thống.
4. **Context loss.** Nhiệm vụ dài hơi quên các ràng buộc ở lượt đầu.
5. **Tool misuse.** Gọi đúng công cụ nhưng sai tham số, hoặc gọi hẳn công cụ sai.

Lan truyền mới là thứ chết người. Agent không phân biệt được "tôi hỏng" với "nhiệm vụ bất khả thi" và thường bịa ra thông báo thành công khi gặp lỗi 400 để khép loop (vòng lặp).

### Giảm thiểu: cổng chặn ở từng bước

Đặt cổng kiểm chứng tự động ở từng bước của chuỗi suy luận, đối chiếu căn cứ thực tế với trạng thái môi trường. Cụ thể:

- Bộ phân loại an toàn cho từng bước (Bài 21).
- Kiểm tra tham số của lượt gọi công cụ (Bài 06).
- Đối chiếu nội dung truy hồi với dữ kiện đã biết (Bài 05, CRITIC).
- Phát hiện ảo giác thành công bằng cách dò lại trạng thái (file có thực sự được tạo không?).

### Giám sát kiểu hỏng hay sai ở đâu

- **Chỉ gắn nhãn khi crash.** Phần lớn lỗi agent vẫn tạo ra đầu ra trông hợp lệ. Cần kiểm tra ở mức nội dung.
- **Không có đường cơ sở.** Phát hiện drift cần một trạng thái tốt đã biết gần nhất; thiếu nó thì bạn không thể nói "cái này đang tệ đi".
- **Cảnh báo quá nhiều.** Mỗi lỗi là một trang cảnh báo. Hãy gom cụm và giới hạn tần suất.

```figure
failure-cascade
```

## Tự tay làm

`code/main.py` cài đặt một bộ gán nhãn kiểu hỏng dùng stdlib:

- Một tập trace tổng hợp bao phủ năm kiểu hỏng.
- Hàm phát hiện cho từng kiểu (mẫu chữ ký trên lượt gọi công cụ, đầu ra, hành động lặp).
- Bộ gán nhãn dán nhãn cho từng trace và báo cáo phân bố kiểu hỏng.

Chạy thử:

```
python3 code/main.py
```

Đầu ra: nhãn cho từng trace và phân bố tổng hợp — một bản tái hiện rẻ tiền của thứ mà gom cụm trace của Phoenix phơi ra.

## Áp dụng

- **Phoenix** để gom cụm drift trong môi trường thật (Bài 24).
- **Langfuse** để phát lại phiên + chú thích.
- **Tự viết** cho các chữ ký đặc thù miền mà nền tảng quan sát của bạn không bắt được.

## Đóng gói

`outputs/skill-failure-detector.md` sinh ra các bộ phát hiện kiểu hỏng may đo cho miền của bạn, nối vào kho trace.

## Bài tập

1. Thêm một bộ phát hiện cho "ảo giác thành công": agent trả về thành công nhưng trạng thái đích không đổi.
2. Gán nhãn 100 trace thật từ một sản phẩm bạn đã làm. Kiểu nào chiếm ưu thế? Chi phí sửa nó là bao nhiêu?
3. Cài đặt một metric "cascade radius" (bán kính ảnh hưởng): với một lỗi ở bước N, nó ảnh hưởng bao nhiêu bước hạ nguồn?
4. Đọc 14 failure mode của MASFT. Chọn ba kiểu áp dụng cho sản phẩm của bạn. Viết bộ phát hiện.
5. Nối một bộ phát hiện vào một job CI: fail build nếu >=5% trace gắn một kiểu hỏng nào đó.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| MASFT | "bộ phân loại kiểu hỏng đa tác tử" | Cách phân loại 14 kiểu của Berkeley |
| Cascading error | "lỗi gợn sóng" | Một sai lầm sớm lan qua N bước |
| Context loss | "quên ràng buộc" | Nhiệm vụ dài hơi làm rơi dữ kiện ở lượt đầu |
| Tool misuse | "sai công cụ / sai tham số" | Lượt gọi hợp lệ nhưng gọi sai |
| Success hallucination | "làm giả hoàn thành" | Agent báo thành công trên lỗi 400; trạng thái không đổi |
| Scope creep | "vượt phạm vi" | Agent làm nhiều hơn yêu cầu |
| Instruction-following deviation | "không tuân lệnh" | Bỏ qua system prompt hoặc ràng buộc của người dùng |
| Sub-intention errors | "lỗi kế hoạch" | Bỏ sót, dư thừa, sai thứ tự khi thực thi kế hoạch |

## Đọc thêm

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 failure mode, 3 nhóm
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — sổ đăng ký rủi ro
- [Arize Phoenix](https://docs.arize.com/phoenix) — gom cụm drift trong thực tế
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — khi các mẫu đơn giản hơn giúp tránh hẳn các kiểu hỏng
