# Đánh giá ngữ cảnh dài — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro quảng cáo 10M token ngữ cảnh. Ở 1M token, MRCR 8 kim tụt xuống 26.3%. Quảng cáo ≠ dùng được. Đánh giá long-context (ngữ cảnh dài) cho bạn biết năng lực thật của model mà bạn đang đưa vào vận hành.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 13 (Hỏi đáp), Giai đoạn 5, Bài 23 (Chiến lược chia khối)
**Time:** ~60 phút

## Vấn đề

Bạn có một hợp đồng 200 trang. Model tự nhận có ngữ cảnh 1M token. Bạn dán hợp đồng vào và hỏi: "Điều khoản chấm dứt là gì?" Model trả lời — nhưng trả lời từ trang bìa, vì điều khoản chấm dứt nằm sâu ở 120k token, vượt quá vùng mà model thực sự chú ý tới.

Đây là khoảng cách năng lực ngữ cảnh của năm 2026. Bảng thông số ghi 1M hay 10M. Thực tế chỉ 60-70% trong đó là dùng được, và "dùng được" còn tuỳ bài toán.

- **Truy xuất (một kim trong đống rơm):** gần như hoàn hảo tới mức tối đa được quảng cáo trên các model tiên phong.
- **Đa bước / tổng hợp:** suy giảm mạnh sau ~128k trên hầu hết model.
- **Suy luận trên các dữ kiện rải rác:** bài toán đầu tiên gục ngã.

Đánh giá long-context đo chính các trục này. Bài học gọi tên các benchmark, mỗi cái thực sự đo gì, và cách dựng một bài kiểm tra needle (kim) tuỳ chỉnh cho lĩnh vực của bạn.

## Khái niệm

![Đường cơ sở NIAH, RULER đa nhiệm, LongBench tổng thể](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023) — kim trong đống rơm.** Đặt một dữ kiện ("từ ma thuật là dứa") ở một độ sâu có kiểm soát trong một ngữ cảnh dài. Yêu cầu model truy xuất nó. Quét độ sâu × độ dài. Đây là benchmark ngữ cảnh dài nguyên thuỷ. Các model tiên phong nay đã bão hoà bài này; nó là mốc cần nhưng chưa đủ.

**RULER (Nvidia, 2024).** 13 loại bài tập thuộc 4 nhóm: truy xuất (một khoá / nhiều khoá / nhiều giá trị), truy vết đa bước (theo dõi biến), tổng hợp (tần suất từ thông dụng), hỏi đáp. Độ dài ngữ cảnh cấu hình được (4k đến 128k+). Nó phơi ra những model bão hoà NIAH nhưng gục ở đa bước. Trong bản phát hành 2024, chỉ một nửa trong 17 model nhận 32k+ ngữ cảnh giữ được chất lượng ở 32k.

**LongBench v2 (2024).** 503 câu hỏi trắc nghiệm, ngữ cảnh 8k-2M từ, sáu nhóm bài tập: hỏi đáp một tài liệu, hỏi đáp nhiều tài liệu, học trong ngữ cảnh dài, hội thoại dài, repo code, dữ liệu có cấu trúc dài. Đây là benchmark production cho hành vi ngữ cảnh dài ngoài đời thực.

**MRCR (Multi-Round Coreference Resolution).** Đồng tham chiếu đa lượt ở quy mô lớn. Các biến thể 8 kim, 24 kim, 100 kim. Nó cho thấy một model giữ được bao nhiêu dữ kiện trước khi sự chú ý suy giảm.

**NoLiMa.** "Kim phi từ vựng." Kim và câu hỏi không chia sẻ từ nào theo nghĩa đen; muốn truy xuất phải qua một bước suy luận ngữ nghĩa. Khó hơn NIAH.

**HELMET.** Ghép nhiều tài liệu lại, hỏi một câu lấy từ bất kỳ tài liệu nào. Kiểm tra sự chú ý có chọn lọc.

**BABILong.** Nhúng các chuỗi suy luận bAbI vào trong đống rơm không liên quan. Kiểm tra suy luận trong đống rơm, không chỉ truy xuất.

### Thực ra nên báo cáo những gì

- **Cửa sổ ngữ cảnh (context window) được quảng cáo.** Con số trên bảng thông số.
- **Độ dài truy xuất hiệu dụng (effective context length).** Tỉ lệ NIAH đạt tại một ngưỡng nào đó (ví dụ 90%).
- **Độ dài suy luận hiệu dụng.** Đạt đa bước hoặc tổng hợp tại ngưỡng đó.
- **Đường cong degradation (suy giảm).** Độ chính xác theo độ dài ngữ cảnh, vẽ riêng cho từng loại bài toán.

Hai con số cho bảng thông số của bạn: hiệu dụng khi truy xuất và hiệu dụng khi suy luận. Thường độ dài suy luận hiệu dụng chỉ bằng 25-50% cửa sổ được quảng cáo.

```figure
gx-niah-decay
```

## Tự tay làm

### Bước 1: một NIAH tuỳ chỉnh cho lĩnh vực của bạn

Xem `code/main.py`. Bộ khung:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Quét `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. Vẽ heatmap. Đó là tấm thẻ NIAH cho model mục tiêu của bạn.

### Bước 2: một biến thể nhiều kim

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

Những câu hỏi kiểu "Ba từ ma thuật là gì?" đòi hỏi truy xuất cả ba. Thành công với một kim không dự báo được thành công với nhiều kim.

### Bước 3: truy vết biến đa bước (kiểu RULER)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

Đáp án đòi hỏi nối ba phép gán. Các model tiên phong ở 128k thường tụt xuống độ chính xác 50-70% tại đây.

### Bước 4: LongBench v2 trên stack của bạn

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

Báo cáo độ chính xác theo từng nhóm. Điểm tổng hợp che mất những khác biệt lớn giữa các bài toán.

## Cạm bẫy

- **Chỉ đánh giá bằng NIAH.** Qua được NIAH ở 1M token không nói gì về đa bước. Luôn chạy RULER hoặc một bài đa bước tuỳ chỉnh.
- **Lấy mẫu độ sâu đồng đều.** Nhiều bản cài đặt chỉ kiểm tra depth=0.5. Hãy kiểm tra depth=0, 0.25, 0.5, 0.75, 1.0 — hiệu ứng "lost in the middle" là có thật.
- **Trùng từ vựng với phần đệm.** Nếu kim chia sẻ từ khoá với phần đệm thì truy xuất trở nên tầm thường. Dùng kim kiểu NoLiMa, không trùng từ.
- **Bỏ qua độ trễ.** Prompt 1M token mất 30-120 giây để prefill. Đo thời gian tới token đầu tiên song song với độ chính xác.
- **Con số do nhà cung cấp tự báo.** OpenAI, Google, Anthropic đều công bố điểm của riêng họ. Luôn chạy lại độc lập trên use case của bạn.

## Áp dụng

Stack của năm 2026:

| Tình huống | Benchmark |
|-----------|-----------|
| Kiểm tra nhanh cho yên tâm | NIAH tuỳ chỉnh với 3 độ sâu × 3 độ dài |
| Chọn model cho production | RULER (13 bài) tại độ dài mục tiêu của bạn |
| Chất lượng hỏi đáp ngoài đời | Nhóm con single-doc-QA của LongBench v2 |
| Suy luận đa bước | BABILong hoặc truy vết biến tuỳ chỉnh |
| Hội thoại / đối thoại | MRCR 8 kim tại độ dài mục tiêu của bạn |
| Hồi quy khi nâng cấp model | Bộ khung NIAH + RULER cố định nội bộ, chạy trên mọi model mới |

Quy tắc ngón tay cho production: đừng bao giờ tin một cửa sổ ngữ cảnh cho tới khi bạn có NIAH + 1 bài suy luận ở đúng độ dài bạn định dùng.

## Đóng gói

Lưu thành `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## Bài tập

1. **Dễ.** Dựng một NIAH với 3 độ sâu (0.25, 0.5, 0.75) × 3 độ dài (1k, 4k, 16k). Chạy trên một model bất kỳ. Vẽ tỉ lệ đạt thành heatmap 3×3.
2. **Trung bình.** Thêm biến thể 3 kim. Đo truy xuất cả 3 kim ở từng độ dài. So với tỉ lệ đạt của một kim ở cùng độ dài.
3. **Khó.** Dựng một bài truy vết biến (X1 → X2 → X3, 3 bước) nhúng trong 64k phần đệm. Đo độ chính xác trên 3 model tiên phong. Báo cáo độ dài suy luận hiệu dụng của từng model.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| NIAH | Needle in haystack | Cắm một dữ kiện vào phần đệm, yêu cầu model truy xuất nó. |
| RULER | NIAH on steroids | 13 loại bài tập thuộc truy xuất / đa bước / tổng hợp / hỏi đáp. |
| Effective context | The real capacity | Độ dài mà tại đó độ chính xác vẫn giữ trên ngưỡng. |
| Lost in the middle | Thiên lệch vị trí (position bias) | Model chú ý chưa đủ tới nội dung nằm giữa các đầu vào dài. |
| Multi-needle | Many facts at once | Nhiều kim cùng lúc; kiểm tra khả năng tung hứng sự chú ý, không chỉ truy xuất. |
| MRCR | Multi-round coref | Đồng tham chiếu 8, 24 hay 100 kim; phơi ra độ bão hoà của sự chú ý. |
| NoLiMa | Non-lexical needle | Kim và câu hỏi không chia sẻ token literal nào; đòi hỏi suy luận. |

## Đọc thêm

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — repo NIAH nguyên thuỷ.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — benchmark đa nhiệm.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — đánh giá ngữ cảnh dài ngoài đời thực.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — kim khó hơn.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — suy luận trong đống rơm.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — bài báo về thiên lệch độ sâu.

---

[← Bài trước: Đánh giá LLM](../../27-llm-evaluation-frameworks/docs/vi.md) · [Bài sau: Dialogue State Tracking (theo dõi trạng thái hội thoại) →](../../29-dialogue-state-tracking/docs/vi.md)
