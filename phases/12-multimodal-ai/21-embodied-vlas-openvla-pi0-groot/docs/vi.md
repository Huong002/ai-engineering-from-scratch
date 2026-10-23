# VLA có thân thể: RT-2, OpenVLA, π0, GR00T

> Lần đầu tiên một mô hình đọc công thức nấu ăn từ một trang web và thực hiện nó trong một robot nhà bếp là RT-2 (Google DeepMind, tháng 7 2023). RT-2 rời rạc hoá hành động thành token văn bản, co-fine-tune một VLM trên dữ liệu web cộng dữ liệu hành động robot, và chứng minh rằng tri thức thị giác-ngôn ngữ quy mô web chuyển giao được sang điều khiển robot. OpenVLA (tháng 6 2024) đưa ra bản tham chiếu 7B mở. Dòng π0 của Physical Intelligence (2024-2025) thêm các action expert flow-matching. GR00T N1 của NVIDIA (tháng 3 2025) cung cấp điều khiển hai hệ (Hệ 1 / Hệ 2) cho robot hình người ở quy mô lớn. Nguyên thuỷ VLA (thị giác-ngôn ngữ-hành động, vision-language-action), một mô hình duy nhất vừa nhìn, vừa đọc, vừa hành động, là cầu nối giữa các mô hình hiểu biết của giai đoạn này và các hệ tự hành ở Giai đoạn 15.

**Type:** Learn
**Languages:** Python (stdlib, action tokenizer + VLA inference skeleton)
**Prerequisites:** Giai đoạn 12, Bài 05 (LLaVA), Giai đoạn 15 (Hệ tự hành, chỉ tham chiếu)
**Time:** ~180 phút

## Mục tiêu học tập

- Mô tả action tokenization: mã hoá bin rời rạc (RT-2), token hành động hiệu quả của FAST, hành động flow-matching liên tục (π0).
- Giải thích vì sao co-fine-tuning trên dữ liệu web + robot giữ được khả năng chuyển giao kiến thức tổng quát sang tác vụ mới.
- So sánh OpenVLA (7B Llama+VLM mở), π0 (flow-matching) và GR00T N1 (hai hệ) trên cùng một tác vụ robot.
- Kể tên bộ dữ liệu Open X-Embodiment và vai trò của nó như kho huấn luyện RT-X.

## Vấn đề

Robot làm việc nhà theo chỉ dẫn ngôn ngữ tự nhiên đã là mục tiêu nghiên cứu từ thập niên 1970. Câu trả lời của thập niên 2020: một mô hình VLA. Cùng kiến trúc VLM dùng cho VQA, nhưng đầu ra là hành động (mô-men khớp, tư thế bộ tác động cuối, lệnh rời rạc) thay vì văn bản.

Những thách thức riêng của VLA:

1. Không gian hành động là liên tục (góc khớp, lực) và nhiều chiều (tay 7-DOF + kẹp 3-DOF = 10 chiều ở 30 Hz).
2. Dữ liệu huấn luyện riêng cho từng robot khan hiếm. Open X-Embodiment có ~1M quỹ đạo (trajectory); văn bản-ảnh web là 5B+.
3. Tần số điều khiển quan trọng. Vòng điều khiển 30 Hz nghĩa là ngân sách 33ms cho mỗi hành động.
4. An toàn. Một hành động sai làm hỏng phần cứng, con người, hoặc tài sản.

## Khái niệm

### Token hoá hành động (RT-2)

Chiêu của RT-2: biểu diễn mỗi giá trị đích của khớp thành một token văn bản đã lượng tử hoá. Chia dải [-1, 1] đã chuẩn hoá thành 256 bin, ánh xạ mỗi bin sang một ID từ vựng. Một hành động 10-DOF thành 10 token ở mỗi bước điều khiển.

Co-fine-tune một VLM PaLM-X trên một hỗn hợp:

- Cặp ảnh-văn bản web (captioning, VQA).
- Minh hoạ robot, hành động dưới dạng token.

Mô hình thấy "pick up the red cube" (ngôn ngữ) → ảnh (thị giác) → chuỗi 10 token hành động (đích khớp đã rời rạc hoá). Tiền huấn luyện web giữ được khả năng chuyển giao kiến thức tổng quát: RT-2 có thể làm theo "move towards the fast-moving object" dù "fast-moving" không có trong dữ liệu huấn luyện.

Suy luận 3-5 Hz trong bài báo RT-2, bị giới hạn bởi giải mã tự hồi quy của VLM.

### OpenVLA — bản tham chiếu 7B mở

OpenVLA (Kim et al., tháng 6 2024) là bản tương đương RT-2 với trọng số mở. Xương sống Llama 7B, encoder thị giác kép DINOv2 + SigLIP, action tokenization trên 256 bin.

Huấn luyện trên Open X-Embodiment (970k quỹ đạo trên 22 robot). Kèm hỗ trợ fine-tuning LoRA để thích ứng với robot mới.

Suy luận: 4-5 Hz trên A100 có lượng tử hoá. Đủ nhanh cho thao tác robot (robot manipulation) chậm, chưa đủ cho điều khiển tần số cao.

### Tokenizer FAST — giải mã hành động nhanh hơn

Pertsch et al. (2024) chỉ ra rằng token hoá theo bin rời rạc là kém hiệu quả — phần lớn hành động tập trung trong một vùng nhỏ của không gian bin. FAST (Frequency-domain Action Sequence Tokenizer) nén chuỗi hành động bằng DCT rồi lượng tử hoá các hệ số.

Một quỹ đạo hành động 30 bước thành ~10 token FAST thay vì 300 token bin rời rạc. Suy luận nhanh hơn 3-5x mà không mất chất lượng.

### π0 và hành động flow-matching

π0 của Physical Intelligence (Black et al., tháng 10 2024) thay token hành động rời rạc bằng một action expert flow-matching:

- Một transformer hành động nhỏ đọc trạng thái ẩn của VLM và xuất chuỗi hành động liên tục 50 bước qua rectified flow.
- Đầu hành động huấn luyện với loss flow-matching; tiền huấn luyện VLM giữ nguyên.
- Suy luận: toàn bộ chuỗi hành động phát ra trong ~5 bước khử nhiễu, tương đương điều khiển 50 Hz.

Tuyên bố của π0: vượt OpenVLA và Octo trên một bộ lớn các tác vụ thao tác. Cách hình thức hoá hành động liên tục giữ được độ mượt mà rời rạc hoá phá hỏng.

π0.5 và π0-FAST là các nâng cấp tăng dần. π0-FAST kết hợp token hoá FAST với flow matching.

### GR00T N1 — hai hệ cho robot hình người

GR00T N1 của NVIDIA (tháng 3 2025) được xây cho robot hình người (>30 DOF, toàn thân):

- Hệ 2: một VLM lớn đọc cảnh + chỉ dẫn, tạo mục tiêu con mức cao ở ~1 Hz.
- Hệ 1: một transformer đầu hành động nhỏ tạo lệnh khớp mức thấp 50-100 Hz có điều kiện theo các mục tiêu con.

Cách tách này ánh xạ tư duy nhanh-và-chậm của Kahneman: Hệ 2 lập kế hoạch, Hệ 1 hành động. Lợi ích: việc lập kế hoạch chậm cỡ VLM không chặn điều khiển nhanh; Hệ 1 vẫn nhỏ để giảm độ trễ.

GR00T N1.7 (cuối 2025) cải thiện khả năng mở rộng dữ liệu. GR00T fine-tune với dữ liệu sim-to-real từ Omniverse.

### Open X-Embodiment

Dữ liệu huấn luyện. RT-X (tháng 10 2023) tập hợp 22 bộ dữ liệu phủ 1M quỹ đạo trên 22 robot. Open X-Embodiment là kho dữ liệu ai cũng dùng:

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table.
- Mỗi mẫu: (trạng thái robot, khung nhìn camera, chỉ dẫn, chuỗi hành động).
- Vệ sinh huấn luyện: thống nhất không gian hành động, chuẩn hoá dải khớp, đổi kích thước camera.

OpenVLA và π0 huấn luyện trên Open X-Embodiment. Khoảng cách miền với bất kỳ robot cụ thể nào được lấp bằng fine-tuning LoRA trên 100-1000 demo riêng cho tác vụ.

### Co-fine-tuning so với chỉ huấn luyện trên robot

Co-fine-tuning trộn dữ liệu VQA web với quỹ đạo robot. Tỉ lệ quan trọng: quá nhiều VQA thì mô hình quên hành động; quá nhiều dữ liệu robot thì mô hình mất kiến thức tổng quát.

Tỉ lệ của RT-2: ~1:1. OpenVLA: ~0.5:1 web-robot. π0: tương tự. Tỉ lệ chính xác là một siêu tham số cần tinh chỉnh theo kích thước bộ dữ liệu.

Huấn luyện chỉ trên robot tạo ra các mô hình đặc thù tác vụ, thất bại với những chỉ dẫn ngoài phân phối. Co-fine-tuning là khác biệt giữa "pick up the red cube (in demo)" và "pick up the third largest object from the left (novel phrasing)."

### An toàn và giới hạn hành động

Mọi VLA đưa vào sản xuất đều kèm theo:

- Giới hạn khớp cứng (không thể tạo mô-men vượt thông số danh định).
- Giới hạn vận tốc (cắt mềm).
- Ranh giới vùng làm việc (bộ tác động cuối không thể rời khỏi bàn).
- Phê duyệt có con người trong vòng lặp cho tác vụ mới.

Những thứ này nằm ngoài VLA như các kiểm tra ở tầng điều khiển. Đầu ra của VLA là một đề xuất, không phải một mệnh lệnh.

```figure
mm-action-tokens
```

## Áp dụng

`code/main.py`:

- Cài đặt action tokenization 256 bin và giải token hoá.
- Phác thảo một tokenizer FAST dựa trên DCT + lượng tử hoá.
- So sánh số token mỗi bước hành động giữa (bin rời rạc, FAST, flow liên tục).
- In bản tóm tắt dòng dõi RT-2 → OpenVLA → π0 → GR00T.

## Đóng gói

Bài này tạo ra `outputs/skill-vla-action-format-picker.md`. Cho một tác vụ robot (thao tác, điều hướng, toàn thân hình người), chọn giữa bin rời rạc + RT-2, FAST + OpenVLA, flow-matching + π0, hoặc hai hệ + GR00T.

## Bài tập

1. Một tay 10-DOF ở tần số điều khiển 30 Hz. Token hoá bin rời rạc với 256 bin phát ra bao nhiêu token mỗi giây? Một VLM 7B có theo kịp không?

2. Token hoá FAST nén quỹ đạo 30 bước xuống còn ~10 token. Người dùng mất gì nếu quỹ đạo có chuyển động tần số cao (ví dụ: đánh trống)?

3. Đầu flow-matching của π0 khử nhiễu trong ~5 bước. So sánh thông lượng với giải mã tự hồi quy 4-5 Hz của OpenVLA.

4. Cách tách Hệ 1 / Hệ 2 của GR00T ánh xạ tới Kahneman. Đề xuất một cách tách khác (Hệ 3?) có thể giúp việc đi bộ bằng hai chân.

5. Đọc mục 4 của Open X-Embodiment về tuyển chọn dữ liệu. Kể tên ba quy tắc tuyển chọn giúp ngăn rò rỉ miền.

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| VLA | "Vision-language-action" | Mô hình nhận ảnh + chỉ dẫn và xuất lệnh hành động |
| Action tokenization | "Discrete bins" | Lượng tử hoá đích khớp liên tục thành 256 bin mỗi chiều, mỗi bin là một ID từ vựng |
| FAST tokenizer | "Frequency action tokens" | DCT + lượng tử hoá để nén quỹ đạo 30 bước xuống ~10 token |
| Co-fine-tune | "Mix web + robot" | Huấn luyện trên dữ liệu VQA web song song với demo robot để giữ kiến thức tổng quát |
| Flow-matching action head | "π0 continuous output" | Transformer nhỏ xuất chuỗi hành động 50 bước qua rectified flow |
| System 1 / System 2 | "Dual-system control" | VLM lớn lập kế hoạch chậm, đầu hành động nhỏ hành động nhanh; mẫu của GR00T |
| Open X-Embodiment | "RT-X dataset" | Bộ dữ liệu 1M quỹ đạo xuyên robot; kho huấn luyện |

## Đọc thêm

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)

---

[← Bài trước: Omni Models: Qwen2.5-Omni và cách tách Thinker-Talker](../../20-omni-models-thinker-talker/docs/vi.md) · [Bài sau: Hiểu tài liệu và sơ đồ →](../../22-document-diagram-understanding/docs/vi.md)
