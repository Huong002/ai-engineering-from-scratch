# Chatbot — Từ dựa trên luật đến mạng nơ-ron đến tác tử LLM

> ELIZA trả lời bằng khớp mẫu. DialogFlow ánh xạ các intent. GPT trả lời từ trọng số. Claude chạy công cụ và kiểm chứng. Mỗi kỷ nguyên đều giải quyết thất bại tệ nhất của kỷ nguyên trước.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 13 (Hỏi đáp), Giai đoạn 5, Bài 14 (Truy hồi thông tin)
**Time:** ~75 phút

## Vấn đề

Một người dùng nói "Tôi muốn đổi chuyến bay." Hệ thống phải xác định họ muốn gì, thiếu thông tin nào, lấy thông tin đó ở đâu và hoàn tất hành động ra sao. Rồi người dùng nói "khoan, nếu tôi huỷ thì sao?" và hệ thống phải nhớ ngữ cảnh, chuyển sang việc khác và giữ nguyên trạng thái.

Hội thoại là bài toán khó với một hệ thống ML. Đầu vào mở, không giới hạn. Đầu ra phải mạch lạc qua nhiều lượt. Hệ thống có thể phải tác động lên thế giới thực (đổi chuyến bay, trừ tiền thẻ). Mọi bước sai đều hiện ra trước mắt người dùng.

Kiến trúc chatbot (trợ lý hội thoại tự động) đã xoay vòng qua bốn paradigm (mô hình). Mỗi mô hình ra đời vì mô hình trước thất bại quá lộ liễu. Bài này đi qua chúng theo thứ tự. Bức tranh triển khai năm 2026 là sự lai ghép của hai mô hình cuối.

## Khái niệm

![Tiến hoá chatbot: dựa trên luật → truy hồi → neural → agent](../assets/chatbot.svg)

### Nửa thế kỷ chạy kịch bản, 1950-2001

Mô hình đầu tiên không kéo dài năm năm. Nó kéo dài năm mươi năm. Biết được vòng đời của nó rất quan trọng, vì mọi hệ thống trong đó đều là cùng một cỗ máy — khớp đầu vào, phát ra một câu trả lời đóng hộp, cập nhật một chút trạng thái — và năm mươi năm thêm luật vào cỗ máy đó không bao giờ tạo ra được trường hợp tổng quát. Chính cái trần đó là lý do các mô hình hai đến bốn tồn tại.

**1950.** Turing lách câu hỏi "máy có thể suy nghĩ không?" bằng một tiêu chí thay thế mang tính vận hành: nếu người thẩm vấn không phân biệt được máy với người qua máy đánh chữ, thì câu hỏi triết học trở nên vô nghĩa. Hội thoại trở thành thước đo chuẩn của cả lĩnh vực trước khi lĩnh vực đó có tên.

**1956.** Cái tên xuất hiện — một hội thảo mùa hè tại Dartmouth đặt ra cụm "trí tuệ nhân tạo" dựa trên phỏng đoán rằng mọi đặc điểm của trí tuệ "về nguyên tắc có thể được mô tả chính xác đến mức có thể chế tạo một cỗ máy mô phỏng nó." Bản đề xuất dự trù hai tháng cho tiến bộ đáng kể.

**1966.** ELIZA ra mắt với mẹo phản chiếu mà bạn sẽ tự dựng ở Bước 1: các luật phân rã rút mảnh từ đầu vào, các luật lắp ghép lặp lại chúng dưới dạng câu hỏi. Tổng cộng khoảng 200 mẫu, không trạng thái, không hiểu biết — vậy mà người dùng vẫn tâm sự với nó. Weizenbaum dành phần còn lại sự nghiệp để lo lắng vì nó cần quá ít máy móc.

**1972.** PARRY, xây tại Stanford để mô hình hoá chứng hoang tưởng, thêm thứ mà ELIZA thiếu: trạng thái bên trong. Các biến số cho sợ hãi, giận dữ và ngờ vực cập nhật sau mỗi lượt và quyết định kịch bản nào chạy tiếp, nên cùng một đầu vào cho ra câu trả lời khác nhau tuỳ vào hội thoại trước đó. Trong một thử nghiệm bản ghi mù, các bác sĩ tâm thần phân biệt PARRY với bệnh nhân người ở mức ngẫu nhiên. Nó là tổ tiên trực tiếp của persona conditioning (điều kiện hoá nhân cách) — một system prompt hiện thực hoá bằng ba số thực. Cùng năm đó, hai con bot được trỏ vào nhau qua ARPANET: một kịch bản trị liệu phỏng vấn một state machine (máy trạng thái) hoang tưởng, cuộc hội thoại bot-với-bot đầu tiên trên một mạng.

**1995.** ALICE mở rộng công thức ELIZA bằng AIML, một phương ngữ XML cho các cặp mẫu-khuôn. Khoảng 40.000 category viết tay, ba lần thắng giải Loebner. Nó chứng minh quy luật mở rộng của hệ thống rule-based (dựa trên luật): thêm luật thì mua được độ phủ, không bao giờ mua được tính tổng quát. Mỗi luật là một khoản nợ mà ai đó phải bảo trì.

**2001.** SmarterChild đưa công thức đó tới 30 triệu người dùng nhắn tin tức thời và thêm truy vấn backend — thời tiết, chứng khoán, giờ chiếu phim — ghép vào khuôn câu. Nheo mắt lại thì nó chính là tool calling trong bộ đồ năm 2001: phân tích intent (ý định), gọi một dịch vụ, đưa kết quả vào câu trả lời.

Năm mươi năm, một cơ chế, số luật ngày càng tăng. Mô hình này kết thúc không phải vì ai bác bỏ nó, mà vì chi phí bảo trì máy trạng thái viết tay tăng tuyến tính theo độ phủ, trong khi kỳ vọng của người dùng tăng theo bất cứ thứ gì họ thấy tuần trước đó.

```figure
chatbot-lineage
```

**Rule-based (dựa trên luật) — ELIZA, AIML, DialogFlow.** Các mẫu do con người viết tay khớp đầu vào người dùng và sinh ra câu trả lời. Các bộ phân loại ý định định tuyến tới những luồng định sẵn. Máy trạng thái điền khuyết thông tin (slot filling) thu thập các thông tin cần thiết. Hoạt động cực tốt trong đúng phạm vi hẹp mà nó được thiết kế. Thất bại ngay khi ra khỏi phạm vi đó. Vẫn được triển khai trong các lĩnh vực an toàn nghiêm ngặt (xác thực ngân hàng, đặt vé máy bay), nơi ảo giác không được dung thứ.

**Retrieval-based (dựa trên truy hồi).** Một hệ thống kiểu FAQ. Mã hoá từng cặp (câu nói, câu trả lời). Lúc chạy, mã hoá tin nhắn của người dùng rồi truy hồi câu trả lời đã lưu gần nhất. Cứ nghĩ tới tính năng "bài viết tương tự" kinh điển của Zendesk. Xử lý cách nói khác ý tốt hơn luật. Không sinh văn bản nên không ảo giác.

**Neural (seq2seq).** Bộ mã hoá-giải mã huấn luyện trên log hội thoại. Sinh câu trả lời từ đầu. Trôi chảy nhưng dễ cho ra câu trả lời chung chung ("Tôi không biết") và lệch sự thật. Không bao giờ thực sự bám chủ đề. Đó là lý do Google, Facebook và Microsoft đều có chatbot gây thất vọng giai đoạn 2016-2019.

**LLM agents (tác tử LLM).** Một mô hình ngôn ngữ được bọc trong vòng lặp biết lập kế hoạch, gọi công cụ và kiểm chứng kết quả. Không phải một chatbot với prompt dài. Một vòng lặp tác tử: lập kế hoạch → gọi công cụ → quan sát kết quả → quyết định bước tiếp. Grounding ưu tiên truy hồi (RAG) giữ nó khỏi ảo giác. Các lệnh gọi công cụ cho phép nó thực sự làm việc. Đây là kiến trúc của năm 2026.

Bốn mô hình không phải là sự thay thế tuần tự. Một chatbot triển khai năm 2026 định tuyến qua cả bốn: dựa trên luật cho xác thực và hành động phá huỷ, truy hồi cho FAQ, sinh bằng mạng nơ-ron cho cách diễn đạt tự nhiên, tác tử LLM cho các truy vấn mở mơ hồ.

## Tự tay làm

### Bước 1: khớp mẫu dựa trên luật

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA trong 20 dòng. Mẹo phản chiếu ("Tôi thấy buồn" → "Vì sao bạn thấy buồn") là màn trình diễn nhà trị liệu tâm lý kinh điển của Weizenbaum 1966. Vẫn còn giá trị để học.

### Bước 2: dựa trên truy hồi (FAQ)

Đoạn minh hoạ này cần `pip install sentence-transformers` (kéo theo cả torch). File `code/main.py` chạy được của bài này dùng độ tương đồng Jaccard thuần stdlib thay thế, để bài học chạy mà không cần phụ thuộc bên ngoài.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

Từ chối dựa trên ngưỡng là lựa chọn thiết kế then chốt. Nếu kết quả khớp tốt nhất không đủ gần, trả về `None` để hệ thống escalation (chuyển sang người thật).

### Bước 3: sinh bằng mạng nơ-ron (baseline)

Dùng một encoder-decoder nhỏ đã tinh chỉnh theo chỉ dẫn (FLAN-T5) hoặc một mô hình hội thoại đã fine-tune. Năm 2026, dùng riêng nó thì không dùng được trong sản xuất (mâu thuẫn, lạc đề, vô nghĩa về sự thật), nhưng nó vẫn nằm trong các hệ thống lai để tạo cách diễn đạt tự nhiên. Các mô hình chỉ có giải mã kiểu DialoGPT cần dấu phân cách lượt rõ ràng và xử lý EOS để cho ra câu trả lời mạch lạc; pipeline text2text của FLAN-T5 chạy được ngay cho một ví dụ giảng dạy.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Bước 4: vòng lặp tác tử LLM

Hình dạng triển khai năm 2026:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

Ba thứ cần gọi tên. Công cụ là các hàm có thể gọi được mà LLM có thể kích hoạt. Vòng lặp dừng khi LLM trả về câu trả lời cuối cùng thay vì một lệnh gọi công cụ. Ngân sách bước ngăn vòng lặp vô hạn trên các nhiệm vụ mơ hồ.

Sản xuất thực tế bổ sung: grounding ưu tiên truy hồi (chèn tài liệu liên quan trước mỗi lần gọi LLM), guardrail (từ chối hành động phá huỷ khi chưa xác nhận), observability (ghi log mọi bước) và đánh giá (các kiểm tra tự động rằng hành vi của tác tử vẫn đúng đặc tả).

### Bước 5: định tuyến lai

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

Mẫu hình: luật tất định cho mọi thứ mang tính phá huỷ, truy hồi cho các FAQ đóng hộp, tác tử LLM cho phần còn lại. Đây là thứ được triển khai trong hệ thống chăm sóc khách hàng năm 2026.

## Áp dụng

Bộ công nghệ năm 2026:

| Trường hợp sử dụng | Kiến trúc |
|---------|---------------|
| Đặt chỗ, thanh toán, xác thực | Máy trạng thái dựa trên luật + điền khuyết thông tin |
| FAQ chăm sóc khách hàng | Truy hồi trên tập câu trả lời đã tuyển chọn |
| Chat tư vấn mở | Tác tử LLM với RAG + gọi công cụ |
| Công cụ nội bộ / trợ lý IDE | Tác tử LLM với gọi công cụ (tìm, đọc, ghi) |
| Chatbot đồng hành / nhập vai nhân vật | LLM đã tinh chỉnh với system prompt nhân cách, truy hồi trên tri thức |

Luôn dùng định tuyến lai trong sản xuất. Không kiến trúc đơn lẻ nào xử lý tốt mọi yêu cầu. Bản thân lớp định tuyến thường là một bộ phân loại ý định nhỏ.

## Những kiểu thất bại vẫn được triển khai

- **Bịa đặt đầy tự tin.** Tác tử LLM tuyên bố đã hoàn thành một hành động mà thực ra chưa. Giảm thiểu: kiểm chứng kết quả, ghi log các lệnh gọi công cụ, không bao giờ để LLM nói nó đã làm gì đó mà không có kết quả trả về thành công từ công cụ.
- **Prompt injection (tiêm prompt).** Người dùng chèn văn bản ghi đè system prompt. Xếp hạng LLM01 trong OWASP Top 10 cho Ứng dụng LLM 2025. Hai dạng: injection trực tiếp (dán vào khung chat) và injection gián tiếp (ẩn trong tài liệu, email hoặc đầu ra công cụ mà tác tử đọc).

  Tỉ lệ tấn công thay đổi theo kịch bản. Tỉ lệ thành công đo được khoảng ~0.5-8.5% trên các mô hình tiên tiến trong benchmark dùng công cụ nói chung và lập trình. Một số cấu hình rủi ro cao cụ thể (tấn công thích ứng nhắm vào tác tử lập trình AI, điều phối có lỗ hổng) đã lên tới ~84%. Các CVE trong sản xuất gồm EchoLeak (CVE-2025-32711, CVSS 9.3) — một lỗ hổng rò rỉ dữ liệu không cần nhấp chuột trong Microsoft 365 Copilot, kích hoạt bằng email do kẻ tấn công kiểm soát.

  Giảm thiểu: coi đầu vào người dùng là không đáng tin trong suốt vòng lặp; làm sạch trước khi gọi công cụ; tách đầu ra công cụ khỏi prompt chính; dùng mẫu Plan-Verify-Execute (PVE) trong đó tác tử lập kế hoạch trước rồi kiểm chứng từng hành động với kế hoạch đó trước khi thực thi (điều này chặn việc kết quả công cụ tiêm các hành động mới ngoài kế hoạch); yêu cầu người dùng xác nhận với hành động phá huỷ; áp dụng đặc quyền tối thiểu cho phạm vi công cụ.

  Không lượng prompt engineering nào loại bỏ hoàn toàn rủi ro này. Cần có các lớp phòng vệ bên ngoài lúc chạy (LLM Guard, kiểm tra allowlist, phát hiện bất thường ngữ nghĩa).
- **Phình phạm vi.** Tác tử lạc nhiệm vụ vì một lệnh gọi công cụ trả về thông tin liên quan lỏng lẻo. Giảm thiểu: thu hẹp hợp đồng công cụ; giữ system prompt tập trung; thêm đánh giá cho tỉ lệ lạc nhiệm vụ.
- **Vòng lặp vô hạn.** Tác tử cứ gọi cùng một công cụ. Giảm thiểu: ngân sách bước, loại trùng lệnh gọi công cụ, LLM judge hỏi "ta có đang tiến bộ không".
- **Cạn cửa sổ ngữ cảnh.** Hội thoại dài đẩy các lượt sớm nhất ra khỏi ngữ cảnh. Giảm thiểu: tóm tắt các lượt cũ, truy hồi các lượt cũ liên quan theo độ tương đồng, hoặc dùng mô hình ngữ cảnh dài.

## Đóng gói

Lưu thành `outputs/skill-chatbot-architect.md`:

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## Bài tập

1. **Easy.** Hiện thực hàm trả lời dựa trên luật ở trên với 10 mẫu cho một bot đặt hàng quán cà phê. Kiểm thử các trường hợp biên: đặt trùng, sửa món, huỷ, ý định không rõ.
2. **Medium.** Xây một hệ lai FAQ + LLM dự phòng. 50 mục FAQ đóng hộp cho một sản phẩm SaaS, LLM dự phòng có truy hồi trên site tài liệu. Đo tỉ lệ từ chối và độ chính xác trên 100 câu hỏi hỗ trợ thật.
3. **Hard.** Hiện thực vòng lặp tác tử ở trên với ba công cụ (search, read-user-data, send-email). Chạy đánh giá với 50 kịch bản kiểm thử, gồm cả các nỗ lực prompt injection. Báo cáo tỉ lệ lạc nhiệm vụ, tỉ lệ nhiệm vụ thất bại và bất kỳ lần injection thành công nào.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Intent | Điều người dùng muốn | Nhãn phân loại (book_flight, reset_password). Được định tuyến tới một handler. |
| Slot | Một mẩu thông tin | Tham số bot cần (ngày, điểm đến). Slot filling là chuỗi các câu hỏi để lấy chúng. |
| RAG | Truy hồi cộng sinh | Truy hồi tài liệu liên quan, rồi neo câu trả lời của LLM vào đó. |
| Tool call | Gọi hàm | LLM phát ra một lệnh gọi có cấu trúc gồm tên + tham số. Runtime thực thi, trả kết quả. |
| Agent loop | Lập kế hoạch, hành động, kiểm chứng | Bộ điều khiển chạy các lần gọi LLM xen kẽ với gọi công cụ cho tới khi xong nhiệm vụ. |
| Prompt injection | Người dùng tấn công prompt | Đầu vào độc hại nhằm ghi đè system prompt. |

## Đọc thêm

- [Turing (1950). Computing Machinery and Intelligence](https://academic.oup.com/mind/article/LIX/236/433/986238) — bài báo biến hội thoại thành thước đo chuẩn của lĩnh vực.
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — bài báo gốc về chatbot dựa trên luật.
- [Colby, Weber, Hilf (1971). Artificial Paranoia](https://doi.org/10.1016/0004-3702(71)90002-6) — kiến trúc biến cảm xúc của PARRY, chatbot có trạng thái đầu tiên.
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — bài báo chatbot mạng nơ-ron muộn của Google, ngay trước khi tác tử LLM tiếp quản.
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — bài báo đặt tên cho mẫu vòng lặp tác tử.
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — hướng dẫn sản xuất năm 2024 vẫn còn đúng vào 2026.
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — bài báo về prompt injection.
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — bảng xếp hạng đưa prompt injection thành mối lo an ninh hàng đầu.
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — các phòng vệ thực tế ở lớp điều phối, gồm Plan-Verify-Execute và luồng người dùng xác nhận.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — CVE rò rỉ dữ liệu không cần nhấp chuột kinh điển từ prompt injection gián tiếp. Trường hợp tham chiếu cho lý do tác tử có quyền ghi cần phòng vệ lúc chạy.

---

[← Bài trước: Sinh văn bản trước Transformer](../../16-text-generation-pre-transformer/docs/vi.md) · [Bài sau: NLP đa ngôn ngữ →](../../18-multilingual-nlp/docs/vi.md)
