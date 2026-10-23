# OpenAI Preparedness Framework và DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2 (tháng 4/2025) đưa ra Research Categories (nhóm nghiên cứu) — Long-range Autonomy, Sandbagging, Autonomous Replication and Adaptation, Undermining Safeguards — tách khỏi Tracked Categories (nhóm theo dõi). Tracked Categories kích hoạt Capabilities Reports và Safeguards Reports, do Safety Advisory Group xem xét. FSF v3 của DeepMind (tháng 9/2025, bổ sung Tracked Capability Levels ngày 17/4/2026) gộp tính tự chủ vào các lĩnh vực ML R&D và Cyber (ML R&D autonomy level 1 = tự động hóa toàn bộ pipeline AI R&D với chi phí cạnh tranh so với người dùng công cụ AI). FSF v3 nêu rõ cách xử lý deceptive alignment (căn chỉnh giả) bằng giám sát tự động để phát hiện việc lạm dụng instrumental reasoning (lập luận về phương tiện để đạt mục tiêu). Cần nói rõ: Research Categories trong PF v2 (gồm Long-range Autonomy) không tự động kích hoạt biện pháp giảm thiểu; chính sách chỉ nói chúng là biện pháp “tiềm năng”. DeepMind cũng nói giám sát tự động “sẽ không còn đủ về lâu dài” nếu instrumental reasoning mạnh lên.

**Type:** Learn
**Languages:** Python (stdlib, three-framework decision-table diff tool)
**Prerequisites:** Giai đoạn 15, Bài 19 (Anthropic RSP)
**Time:** ~45 phút

## Vấn đề

Bài 19 đã đọc kỹ chính sách mở rộng quy mô của Anthropic. Bài này hoàn thiện bức tranh bằng chính sách của OpenAI và DeepMind. Ba tài liệu cùng giải quyết một câu hỏi — khi nào một phòng lab AI tiên phong nên tạm dừng hoặc đặt điều kiện cho một mô hình — và vừa thống nhất ở vài nhóm năng lực, vừa khác nhau ở những điểm quan trọng.

Điểm chung: cả ba coi long-range autonomy (tính tự chủ dài hạn) là nhóm năng lực cần theo dõi. Cả ba nhìn nhận hành vi đánh lừa (alignment faking, sandbagging) là một loại rủi ro riêng. Cả ba có bộ phận đánh giá nội bộ. Điểm khác: OpenAI chia thành “Tracked” (bắt buộc giảm thiểu) và “Research” (không tự động kích hoạt). DeepMind gộp tính tự chủ vào hai lĩnh vực thay vì đặt tên riêng. Các phòng lab gọi nhóm là Tracked và Research, Critical và Moderate, hay Tier-1 và Tier-2; năng lực thuộc nhóm nào sẽ dẫn đến hành động khác nhau ở mỗi nơi.

Đọc đối chiếu cả ba là bài tập hữu ích. Cùng một năng lực có thể “bắt buộc giảm thiểu” tại Anthropic, “theo dõi nhưng không kích hoạt” tại OpenAI, và “theo dõi trong một lĩnh vực cụ thể” tại DeepMind. Lập trường chính sách rất quan trọng.

## Khái niệm

### OpenAI Preparedness Framework v2 (tháng 4/2025)

Cấu trúc:

- **Tracked Categories**: kích hoạt Capabilities Reports (mô hình làm được gì) và Safeguards Reports (đã có biện pháp bảo vệ nào). Safety Advisory Group xem xét trước khi triển khai.
- **Research Categories**: các năng lực mô hình được phòng lab theo dõi nhưng chưa cam kết biện pháp giảm thiểu cụ thể. Gồm Long-range Autonomy, Sandbagging, Autonomous Replication and Adaptation, Undermining Safeguards.

Điểm mấu chốt của v2: Research Categories không tự động kích hoạt biện pháp giảm thiểu. Chính sách chỉ nói đến biện pháp “tiềm năng”. Đây là khác biệt cần chú ý: một nhóm thuộc Tracked hay Research quyết định nó được “đặt điều kiện” hay chỉ “quan sát”.

### DeepMind Frontier Safety Framework v3 (tháng 9/2025; bổ sung Tracked Capability Levels ngày 17/4/2026)

Cấu trúc:

- **Critical Capability Levels (CCLs)**: các ngưỡng năng lực trong năm lĩnh vực: Cyber, Bio, ML R&D, CBRN, Autonomy (được gộp vào ML R&D và Cyber).
- **Tracked Capability Levels**: các mức chi tiết hơn, bổ sung vào tháng 4/2026. Ví dụ cụ thể: ML R&D autonomy level 1 = tự động hóa toàn bộ pipeline AI R&D với chi phí cạnh tranh so với người dùng công cụ AI.
- **Giám sát deceptive alignment**: cam kết rõ ràng về giám sát tự động để phát hiện việc lạm dụng instrumental reasoning.

Cách nhìn về tính tự chủ khác OpenAI. DeepMind không giữ “Autonomy” làm lĩnh vực cấp cao nhất mà gộp vào các lĩnh vực nơi nó có thể gây hại (ML R&D và Cyber). Lập luận ủng hộ là tính tự chủ không gắn với lĩnh vực thì chỉ là năng lực, chưa phải rủi ro; lập luận phản biện là tính tự chủ xuyên lĩnh vực là rủi ro bao trùm mà framework cần gọi tên.

### Điểm chung của cả ba

- Safety Advisory Group nội bộ (Anthropic SAG, OpenAI SAG, ủy ban nội bộ DeepMind). Đánh giá trước khi triển khai mô hình có năng lực cao.
- Nêu rõ deceptive alignment / alignment faking là một loại rủi ro.
- Có tài liệu định kỳ theo lịch công bố (Anthropic: Frontier Safety Roadmap, Risk Report; OpenAI: Capabilities and Safeguards Reports; DeepMind: chu kỳ cập nhật FSF).
- Thừa nhận phòng vệ chỉ bằng giám sát có giới hạn. DeepMind nói rõ: “giám sát tự động sẽ không còn đủ về lâu dài”.

### Những điểm khác nhau

- **Anthropic**: bỏ cam kết tạm dừng trong v3.0; ngưỡng AI R&D-4 là điều kiện tiếp theo được nêu đích danh.
- **OpenAI**: chia Tracked và Research; Research Categories (gồm Long-range Autonomy) không tự động đặt điều kiện.
- **DeepMind**: gộp tính tự chủ vào các lĩnh vực khác; Tracked Capability Levels bổ sung mức chi tiết vào tháng 4/2026.

### Sandbagging: năng lực cụ thể gây khó cho cả ba

Sandbagging (mô hình cố tình thể hiện kém trong các bài đánh giá) thuộc Research Categories của OpenAI. RSP v3.0 của Anthropic xử lý qua khoảng cách giữa bối cảnh đánh giá và thực tế (Bài 1). DeepMind xử lý bằng giám sát deceptive alignment trong FSF v3.

Nếu mô hình sandbag trong đánh giá, ngưỡng năng lực của mọi framework đều bị ước tính thấp. Framework chỉ hiệu quả khi phép đo đáng tin. Vì vậy, ngoài tự đánh giá của phòng lab, cần có đo lường độc lập (Bài 21, METR) và đánh giá đối kháng.

### Kỹ năng đọc chính sách

- Tìm: mọi năng lực bạn quan tâm phải tìm được trong chính sách. Nếu không tìm thấy, chính sách không bao quát nó.
- Phân loại: nó thuộc Tracked (kích hoạt giảm thiểu) hay Research (được theo dõi nhưng không kích hoạt)? OpenAI gọi tên hai nhóm này; Anthropic và DeepMind có cách phân loại tương ứng.
- Lịch cập nhật: chính sách được cập nhật theo lịch công bố hay chỉ sau những sự kiện cụ thể? Lịch công bố là cam kết mạnh hơn.
- Tính độc lập: đánh giá bên ngoài là bắt buộc hay tùy chọn? Anthropic hợp tác với Apollo và US AI Safety Institute; OpenAI với METR; DeepMind chủ yếu dựa vào SAG nội bộ.

```figure
a5-tracked-vs-research
```

## Áp dụng

`code/main.py` triển khai công cụ nhỏ để so sánh bảng quyết định. Với một năng lực (tính tự chủ, deceptive alignment, tự động hóa R&D, gia tăng năng lực tấn công mạng, v.v.), công cụ cho biết mỗi chính sách trong ba chính sách phân loại năng lực đó ra sao và kích hoạt biện pháp giảm thiểu nào. Đây là công cụ hỗ trợ đọc, không phải công cụ chính sách.

## Đóng gói

`outputs/skill-cross-policy-diff.md` tạo bản so sánh ba chính sách cho một năng lực cụ thể, dựa trên ba framework.

## Bài tập

1. Chạy `code/main.py`. Xác nhận kết quả của công cụ so sánh khớp với chính sách ở ít nhất hai năng lực mà bạn có thể đối chiếu với tài liệu gốc.

2. Đọc toàn bộ OpenAI Preparedness Framework v2. Xác định từng Research Category. Với mỗi nhóm, viết một câu giải thích vì sao nó thuộc Research thay vì Tracked.

3. Đọc toàn bộ DeepMind FSF v3 và bản cập nhật Tracked Capability Levels tháng 4/2026. Xác định tiêu chí đánh giá cụ thể của ML R&D autonomy level 1. Bạn sẽ đo lường nó từ bên ngoài thế nào?

4. Sandbagging thuộc Research Categories của OpenAI. Thiết kế bài đánh giá buộc mô hình đang sandbag bộc lộ năng lực thực. Tham chiếu phần thảo luận về eval-context-gaming ở Bài 1.

5. So sánh ba chính sách về một năng lực cụ thể (bạn chọn). Nêu chính sách nào phân loại chặt chẽ nhất và chính sách nào kém nhất. Dùng văn bản gốc để lý giải.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|---|---|---|
| Preparedness Framework | “Chính sách mở rộng quy mô của OpenAI” | PF v2 (tháng 4/2025); phân chia Tracked và Research |
| Tracked Category | “Bắt buộc giảm thiểu” | Kích hoạt Capabilities + Safeguards Reports; SAG đánh giá |
| Research Category | “Chỉ theo dõi” | Được theo dõi nhưng không tự động kích hoạt giảm thiểu; gồm Long-range Autonomy |
| Frontier Safety Framework | “Chính sách mở rộng quy mô của DeepMind” | FSF v3 (tháng 9/2025) + Tracked Capability Levels (tháng 4/2026) |
| CCL | “Ngưỡng năng lực trọng yếu” | Ngưỡng của DeepMind theo từng lĩnh vực (Cyber, Bio, ML R&D, CBRN) |
| ML R&D autonomy level 1 | “Tự động hóa R&D” | Tự động hóa toàn bộ pipeline AI R&D với chi phí cạnh tranh |
| Sandbagging | “Cố tình thể hiện kém” | Mô hình thể hiện kém trong đánh giá; thuộc OpenAI Research Categories |
| Instrumental reasoning | “Lập luận về phương tiện và mục tiêu” | Lập luận về cách đạt mục tiêu; đối tượng giám sát của DeepMind |

## Đọc thêm

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — thông báo về v2.
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — tài liệu đầy đủ.
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — thông báo về FSF v3.
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — phần bổ sung Tracked Capability Levels.
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — ví dụ về Risk Report theo định dạng FSF.

---

[← Bài trước: Chính sách mở rộng có trách nhiệm v3.0 của Anthropic](../../19-anthropic-rsp/docs/vi.md) · [Bài sau: Chân trời thời gian METR và đánh giá năng lực độc lập →](../../21-metr-external-evaluation/docs/vi.md)
