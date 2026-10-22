# Dialogue State Tracking (theo dõi trạng thái hội thoại)

> "Tôi muốn một nhà hàng rẻ ở phía bắc... à thôi, đổi sang mức vừa... và thêm món Ý." Ba lượt thoại (turn), ba lần cập nhật trạng thái. DST giữ dict slot (khe thông tin)–value luôn khớp để việc đặt bàn chạy đúng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5, Bài 17 (Chatbot), Giai đoạn 5, Bài 20 (Đầu ra có cấu trúc)
**Time:** ~75 phút

## Vấn đề

Trong một hệ thống hội thoại hướng tác vụ, mục tiêu của người dùng được mã hoá thành một tập cặp slot–value: `{cuisine: italian, area: north, price: moderate}`. Mỗi lượt thoại của người dùng có thể thêm, đổi hoặc xoá một slot. Hệ thống phải đọc toàn bộ cuộc hội thoại và xuất ra trạng thái hiện tại cho đúng.

Sai một slot duy nhất là hệ thống đặt nhầm nhà hàng, xếp nhầm chuyến bay, hoặc trừ tiền nhầm thẻ. DST là bản lề giữa điều người dùng nói và điều backend thực thi.

Vì sao đến năm 2026 nó vẫn quan trọng dù đã có LLM:

- Các lĩnh vực nhạy cảm về tuân thủ (ngân hàng, y tế, đặt vé máy bay) đòi hỏi giá trị slot tất định, không phải văn bản sinh tự do.
- Các agent dùng công cụ vẫn cần phân giải slot trước khi gọi API.
- Sửa ý qua nhiều lượt khó hơn vẻ ngoài: "à không, đổi sang thứ Năm."

Pipeline hiện đại: khái niệm DST cổ điển + bộ trích xuất LLM + rào chắn đầu ra có cấu trúc.

## Khái niệm

![DST: lịch sử hội thoại → trạng thái slot-value](../assets/dst.svg)

**Cấu trúc tác vụ.** Một lược đồ (schema) định nghĩa các domain (nhà hàng, khách sạn, taxi) và các slot của chúng (cuisine, area, price, people). Mỗi slot có thể rỗng, được điền một giá trị từ tập đóng (price: {cheap, moderate, expensive}), hoặc một giá trị tự do (name: "The Copper Kettle").

**Hai cách hình thức hoá DST.**

- **Phân loại.** Với mỗi cặp (slot, candidate_value), dự đoán có/không. Hợp với slot có từ vựng đóng. Chuẩn trước 2020.
- **Sinh.** Cho cả hội thoại, sinh giá trị slot dưới dạng văn bản tự do. Hợp với slot có từ vựng mở. Lựa chọn mặc định hiện đại.

**Thước đo.** Joint Goal Accuracy (độ chính xác mục tiêu chung, JGA) — tỉ lệ các lượt thoại mà *mọi* slot đều đúng. Được ăn cả, ngã về không. Bảng xếp hạng MultiWOZ 2.4 năm 2026 đỉnh ở khoảng 83%.

**Kiến trúc.**

1. **Dựa trên luật (regex theo slot + từ khoá).** Baseline mạnh cho domain hẹp. Dễ debug.
2. **TripPy / BERT-DST.** Sinh dựa trên copy với mã hoá BERT. Chuẩn thời tiền-LLM.
3. **LDST (LLaMA + LoRA).** LLM tinh chỉnh theo chỉ dẫn với prompt domain-slot. Đạt chất lượng ngang ChatGPT trên MultiWOZ 2.4.
4. **Không cần ontology (bản thể) (2024–26).** Bỏ qua lược đồ; sinh thẳng tên slot và giá trị. Xử lý được domain mở.
5. **Prompt + đầu ra có cấu trúc (2024–26).** LLM với lược đồ Pydantic + giải mã ràng buộc. 5 dòng code, sẵn sàng production.

### Các kiểu thất bại kinh điển

- **Đồng tham chiếu xuyên lượt.** "Cứ giữ lựa chọn đầu tiên nhé." Cần xác định đó là lựa chọn nào.
- **Ghi đè hay nối thêm.** Người dùng nói "thêm món Ý." Bạn thay cuisine hay nối thêm?
- **Xác nhận ngầm.** "OK được đấy" — câu đó có chấp nhận booking đã đề nghị không?
- **Sửa ý.** "À đổi sang 7 giờ tối." Phải cập nhật time mà không xoá các slot khác.
- **Đồng tham chiếu tới phát ngôn trước của hệ thống.** "Ừ, cái đó." "Cái đó" là cái nào?

```figure
n5-slot-tracker
```

## Tự tay làm

### Bước 1: bộ trích xuất slot dựa trên luật

Xem `code/main.py`. Regex + từ điển đồng nghĩa phủ 70% phát ngôn chuẩn trong các domain hẹp:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

Dễ vỡ ngoài từ vựng chuẩn. Chạy tốt cho các xác nhận slot tất định.

### Bước 2: vòng lặp cập nhật trạng thái

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

Ba bất biến:

- Không bao giờ reset slot mà người dùng không chạm tới.
- Phủ định rõ ràng ("thôi bỏ cuisine") phải xoá.
- Sửa ý của người dùng ("à...") phải ghi đè, không nối thêm.

### Bước 3: DST do LLM dẫn dắt với đầu ra có cấu trúc

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic đảm bảo một object trạng thái hợp lệ. Không regex, không lệch lược đồ, không slot bịa ra.

### Bước 4: đánh giá JGA

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

Hiệu chuẩn: hệ thống đúng TẤT CẢ slot ở bao nhiêu phần trăm lượt thoại? Với MultiWOZ 2.4, các hệ thống đỉnh năm 2026: 80-83%. Hệ thống trong domain của bạn nên vượt mức đó trên từ vựng hẹp, nếu không baseline LLM sẽ thắng bạn.

### Bước 5: xử lý sửa ý

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

Khi phát hiện sửa ý, ghi đè slot được cập nhật gần nhất thay vì nối thêm. Khó làm đúng nếu không có LLM hỗ trợ. Mẫu hiện đại: luôn để LLM sinh lại toàn bộ trạng thái từ lịch sử thay vì cập nhật tăng dần — cách này tự nhiên xử lý được sửa ý.

## Cạm bẫy

- **Chi phí sinh lại toàn bộ lịch sử.** Để LLM sinh lại trạng thái mỗi lượt tốn tổng O(n²) token. Giới hạn lịch sử hoặc tóm tắt các lượt cũ.
- **Lược đồ trôi.** Thêm slot mới về sau làm hỏng dữ liệu huấn luyện cũ. Hãy đánh version cho lược đồ.
- **Phân biệt hoa thường.** "Italian" vs "italian" vs "ITALIAN" — chuẩn hoá mọi nơi.
- **Kế thừa ngầm.** Nếu trước đó người dùng đã nói "cho 4 người," một yêu cầu mới về giờ khác không được xoá people. Luôn truyền toàn bộ lịch sử.
- **Tự do vs tập đóng.** Tên, giờ và địa chỉ cần slot tự do; cuisine và area là đóng. Trộn cả hai trong lược đồ.

## Áp dụng

Stack năm 2026:

| Tình huống | Cách tiếp cận |
|-----------|----------|
| Domain hẹp (một hoặc hai ý định) | Dựa trên luật + regex |
| Domain rộng, có dữ liệu gán nhãn | LDST (LLaMA + LoRA trên dữ liệu kiểu MultiWOZ) |
| Domain rộng, không nhãn, sẵn sàng production | LLM + Instructor + lược đồ Pydantic |
| Nói / giọng nói | ASR + bộ chuẩn hoá + LLM-DST |
| Luồng đặt chỗ đa domain | LLM dẫn dắt theo lược đồ (schema-guided) với model Pydantic cho từng domain |
| Nhạy cảm về tuân thủ | Dựa trên luật là chính, LLM dự phòng kèm luồng xác nhận |

## Đóng gói

Lưu thành `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## Bài tập

1. **Dễ.** Xây state tracker dựa trên luật trong `code/main.py` cho 3 slot (cuisine, area, price). Test trên 10 hội thoại tự soạn. Đo JGA.
2. **Vừa.** Cùng bộ dữ liệu đó với Instructor + Pydantic + một LLM nhỏ. So sánh JGA. Xem kỹ các lượt khó nhất.
3. **Khó.** Cài cả hai và định tuyến: dựa trên luật là chính, LLM dự phòng khi cách luật trích được <2 slot với độ tin cậy. Đo JGA kết hợp và chi phí suy luận mỗi lượt.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| DST | Theo dõi trạng thái hội thoại | Giữ dict slot–value qua các lượt thoại. |
| Slot | Đơn vị ý định của người dùng | Tham số có tên mà backend cần (cuisine, date). |
| Domain | Vùng tác vụ | Nhà hàng, khách sạn, taxi — các tập slot. |
| JGA | Joint Goal Accuracy | Tỉ lệ lượt thoại mà mọi slot đều đúng. Được ăn cả, ngã về không. |
| MultiWOZ | Bộ benchmark | Bộ dữ liệu WOZ đa domain; chuẩn đánh giá DST. |
| Ontology-free DST | Không có lược đồ | Sinh thẳng tên slot và giá trị, không có danh sách cố định. |
| Correction | "À thật ra..." | Lượt thoại ghi đè một slot đã được điền trước đó. |

## Đọc thêm

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — bộ benchmark chuẩn.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — tinh chỉnh theo chỉ dẫn LLaMA + LoRA cho DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — công cụ DST dựa trên copy.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — TOD không giám sát dựa trên EM.
- [Bảng xếp hạng MultiWOZ](https://github.com/budzianowski/multiwoz) — kết quả DST chuẩn.

---

[← Bài trước: Đánh giá ngữ cảnh dài](../../28-long-context-evaluation/docs/vi.md)
