# InternVL3: Tiền huấn luyện đa phương thức nguyên bản

> Mọi VLM mở trước InternVL3 đều theo cùng một công thức ba bước: lấy một LLM văn bản đã huấn luyện trên hàng nghìn tỉ token văn bản, gắn thêm một vision encoder (bộ mã hoá thị giác), rồi tinh chỉnh các chỗ nối. Cách này chạy được nhưng để lại alignment debt (món nợ căn chỉnh) — LLM văn bản đã tiêu hết ngân sách tiền huấn luyện cho văn bản thuần và không hiểu visual token (token thị giác) một cách tự nhiên. Khi bạn thêm thị giác sau (post-hoc), LLM phải học lại cách liên hệ đầu vào thị giác với lập luận văn bản mà không quên văn bản. InternVL3 (Zhu et al., tháng 4/2025) bác bỏ cách làm post-hoc: một lần tiền huấn luyện, văn bản và đa phương thức xen kẽ ngay từ bước đầu. Kết quả ngang Gemini 2.5 Pro trên MMMU-Pro với 78B tham số, mở. Bài này đọc lập luận ủng hộ tiền huấn luyện nguyên bản và những gì thay đổi khi bạn làm vậy.

**Type:** Learn
**Languages:** Python (stdlib, training-corpus mixer)
**Prerequisites:** Giai đoạn 12, Bài 05, Giai đoạn 12, Bài 07 (công thức)
**Time:** ~120 phút

## Mục tiêu học tập

- Giải thích vì sao huấn luyện VLM post-hoc tích lũy alignment debt, nêu ba triệu chứng đo được (quên thảm khốc, trôi đáp án, mâu thuẫn thị giác-văn bản).
- Mô tả tỉ lệ trộn tập dữ liệu tiền huấn luyện nguyên bản của InternVL3 và vì sao tỉ lệ text : interleaved : caption quan trọng.
- So sánh V2PE (variable visual position encoding — mã hoá vị trí thị giác biến thiên) với M-RoPE của Qwen2-VL.
- Kể tên các tối ưu triển khai Visual Resolution Router (ViR) và Decoupled Vision-Language (DvD).

## Vấn đề

Huấn luyện VLM post-hoc là mặc định. LLaVA, BLIP-2, Qwen-VL, Idefics — tất cả đều lấy một LLM đã tiền huấn luyện (Llama, Vicuna, Qwen, Mistral) và thêm thị giác. Các giai đoạn huấn luyện thường như sau:

1. LLM đóng băng + vision encoder đóng băng + projector huấn luyện được, huấn luyện trên các cặp caption để căn chỉnh embedding.
2. Mở băng LLM, huấn luyện trên dữ liệu chỉ dẫn (LLaVA-Instruct, ShareGPT4V).
3. Tinh chỉnh theo tác vụ cụ thể (tuỳ chọn).

Ba triệu chứng của alignment debt xuất hiện:

- Quên thảm khốc. VLM post-hoc quên kỹ năng chỉ-văn-bản. Điểm GSM8K giảm 5-10 điểm. Điểm Hellaswag giảm. Agent thuần văn bản bị thoái lui.
- Trôi đáp án. Cùng một câu hỏi thị giác, chỉ cần đổi cách diễn đạt một chút là ra đáp án khác. Vision encoder kết nối với LLM bằng liên kết yếu hơn liên kết giữa các token của chính LLM.
- Mâu thuẫn thị giác-văn bản. VLM có thể mô tả đúng một ảnh rồi trả lời một câu hỏi mâu thuẫn với chính mô tả đó. Token thị giác không tham gia vào các kiểm tra nhất quán nội bộ của LLM giống như văn bản.

Những triệu chứng này được ghi nhận đầy đủ. MM1.5 Mục 4 định lượng chúng. Các thí nghiệm cắt bỏ (ablation) của LLaVA-OneVision gợi tới chúng. Tiền huấn luyện nguyên bản là câu trả lời.

## Khái niệm

### Tiền huấn luyện đa phương thức nguyên bản

InternVL3 huấn luyện từ đầu trên một tập dữ liệu đa phương thức nguyên bản ngay từ bước một. Tỉ lệ trộn là:

- 40% dữ liệu chỉ văn bản (FineWeb, Proof-Pile-2, v.v.)
- 35% dữ liệu ảnh-văn bản xen kẽ (kiểu OBELICS, MMC4)
- 20% dữ liệu cặp ảnh-caption
- 5% dữ liệu video-văn bản

Token thị giác, token văn bản và tương tác liên phương thức đều tham gia cùng một loss ngay từ bước gradient đầu tiên. Không tiền huấn luyện căn chỉnh, không giai đoạn đóng băng projector, không có quên thảm khốc để phải khắc phục.

Huấn luyện là một giai đoạn duy nhất cho mô hình gốc. Tinh chỉnh chỉ dẫn đến sau, nhưng mô hình gốc đã hiểu token thị giác như công dân hạng nhất.

### V2PE (mã hoá vị trí thị giác biến thiên)

Qwen2-VL dùng M-RoPE với phân bổ trục cố định. InternVL3 giới thiệu V2PE: mã hoá vị trí thay đổi theo từng loại phương thức (văn bản, ảnh, video) với scaling học được. Trong thực tế:

- Token văn bản nhận vị trí 1D (chỉ số văn bản).
- Patch ảnh nhận vị trí 2D (hàng, cột).
- Khung video nhận vị trí 3D (thời gian, hàng, cột).

Ba loại chia sẻ cùng cơ số tần số RoPE, nhưng phân bổ chiều ẩn cho mỗi dải là tham số học được thay vì chia cố định. Tự do đánh đổi giữa độ phân giải tần số thời gian và không gian trong lúc tiền huấn luyện.

Tuyên bố từ ablation của V2PE: hơn M-RoPE 1-2 điểm trên các benchmark video với cùng lượng compute. Không phải cuộc cách mạng, nhưng gọn hơn.

### Visual Resolution Router (ViR)

Tối ưu triển khai. Không phải ảnh nào cũng cần mã hoá toàn độ phân giải. Một bức ảnh chỉ có một vật thể, ít chi tiết, sẽ lãng phí token khi mã hoá ở 1280px native resolution (độ phân giải gốc). ViR là một classifier nhỏ dự đoán độ phân giải tối thiểu cần thiết để trả lời câu hỏi, trước khi mã hoá.

Việc định tuyến có ba mức: độ phân giải thấp (256 token), trung bình (576), cao (2048+). Với 60% truy vấn trong lưu lượng sản xuất, mức thấp hoặc trung bình là đủ. Hiệu ứng ròng: thông lượng gấp 2-3 lần với chất lượng tương đương.

### Triển khai tách rời thị giác-ngôn ngữ (DvD)

Khi phục vụ một VLM lớn, vision encoder chạy một lần cho mỗi ảnh nhưng LLM chạy tự hồi quy cho mỗi token đầu ra. Hai thành phần có nút thắt khác nhau (thị giác = băng thông bộ nhớ GPU cho conv + attention; LLM = KV cache). DvD tách chúng lên các GPU riêng với luồng dữ liệu nối giữa.

Với mô hình 8B + encoder 400M, DvD gần như nhân đôi thông lượng mỗi node so với đặt chung.

### Chất lượng một giai đoạn so với nhiều giai đoạn

Tuyên bố benchmark chính của InternVL3: ở 78B tham số, ngang MMMU-Pro của Gemini 2.5 Pro. Ở 38B, ngang GPT-4o. Ở 8B, dẫn đầu bảng xếp hạng open-8B. Tất cả trên một công thức tiền huấn luyện một giai đoạn + tinh chỉnh chỉ dẫn.

Giả thuyết alignment debt đo được: InternVL3-8B mất ít điểm benchmark văn bản (MMLU, GSM8K) hơn Qwen2.5-VL-7B trên mỗi đơn vị lợi ích benchmark thị giác. Mô hình tổng quát hơn vì huấn luyện là một khối, không phải hai.

### InternVL3.5 và InternVL-U

InternVL3.5 (tháng 8/2025) mở rộng công thức. Cùng cách tiền huấn luyện nguyên bản, nhiều dữ liệu hơn, nhiều tham số hơn. Cải thiện MMMU là tăng dần.

InternVL-U (2026) thêm sinh hợp nhất — đầu ra ảnh qua các head MMDiT trên cùng backbone. Chữ "U" là viết tắt của "Understanding + generation" (hiểu + sinh), đuổi theo các mô hình hợp nhất kiểu Transfusion (Bài 12.13). Cùng backbone tiền huấn luyện nguyên bản hỗ trợ cả head hiểu và head sinh.

### Đánh đổi của tiền huấn luyện nguyên bản

Tiền huấn luyện nguyên bản không miễn phí:

- Compute. Huấn luyện một VLM mới từ đầu tốn ngang huấn luyện một LLM văn bản — hàng triệu giờ GPU. Thích nghi post-hoc tận dụng trọng số LLM sẵn có, tiết kiệm phần lớn chi phí.
- Dữ liệu. Tập dữ liệu ảnh-văn bản xen kẽ ở quy mô lớn rất hiếm. OBELICS có 141M tài liệu; MMC4 có 571M. Văn bản thuần có tới 15T token. Khan hiếm dữ liệu tiền huấn luyện đa phương thức là ràng buộc cứng.
- Tái dùng LLM gốc. Tiền huấn luyện nguyên bản từ bỏ lựa chọn thay một LLM mới vào sau này. Post-hoc cho phép bạn đổi Llama-3.1 sang Llama-4 chỉ bằng cách huấn luyện lại adapter.

Cái cược InternVL3 đặt: alignment debt tệ hơn tổn thất tái dùng. Các benchmark ủng hộ tuyên bố đó. Rào cản chi phí ngăn các lab sau này sao chép rẻ. VLM post-hoc vẫn tồn tại vì chúng vẫn rẻ hơn với phần lớn dự án.

```figure
l5-native-pretrain
```

## Áp dụng

`code/main.py` là bộ trộn tập dữ liệu huấn luyện và giả lập router ViR. Nó:

- Nhận tỉ lệ trộn đích (%text, %interleaved, %caption, %video) và tính số bước kỳ vọng cho mỗi phương thức.
- Giả lập định tuyến ViR trên một batch truy vấn (phân bố: 50% ít chi tiết, 30% trung bình, 20% nhiều chi tiết) và báo cáo số token trung bình.
- Báo cáo ước lượng thông lượng DvD với FLOPs của encoder và LLM.
- In so sánh song song giữa post-hoc và tiền huấn luyện nguyên bản về tham số, compute, dữ liệu và các triệu chứng alignment debt kỳ vọng.

## Đóng gói

Bài này tạo ra `outputs/skill-native-vs-posthoc-auditor.md`. Cho một kế hoạch huấn luyện VLM đề xuất, nó kiểm tra xem nên đi theo hướng nguyên bản hay post-hoc, gắn cờ rủi ro alignment debt và đề xuất tỉ lệ trộn dữ liệu. Dùng nó khi bạn đang ước lượng quy mô cho một dự án open-VLM mới và cần chọn chiến lược huấn luyện.

## Bài tập

1. Ước lượng chênh lệch compute giữa InternVL3-8B (tiền huấn luyện nguyên bản) và LLaVA-OneVision-7B (post-hoc). Tỉ lệ giờ GPU xấp xỉ? Điều gì giải thích khoảng cách đó?

2. InternVL3 báo cáo 40% text / 35% interleaved / 20% caption / 5% video. Nếu tác vụ đích của bạn nặng video, đề xuất tỉ lệ mới và lập luận vì sao mô hình gốc vẫn cần nhiều dữ liệu văn bản và caption.

3. Đọc Mục 4 của MM1.5 về quên. Kể tên đúng benchmark mà huấn luyện post-hoc cho thấy thoái lui lớn nhất. Thoái lui đó tốn bao nhiêu?

4. ViR định tuyến 60% lưu lượng sang mã hoá độ phân giải thấp. Những loại truy vấn nào nó định tuyến sai (đẩy sang low-res khi cần high-res)? Đề xuất ba kiểu lỗi của router.

5. DvD tách thị giác và LLM lên các GPU riêng. Với mẫu lưu lượng nào thì DvD làm hại thông lượng thay vì giúp?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Native multimodal pretraining | "Từ đầu, cùng nhau" | Token văn bản + ảnh + video tham gia vào loss từ bước 1, không gắn thêm sau |
| Alignment debt | "Hình phạt post-hoc" | Thoái lui đo được ở kỹ năng văn bản và độ nhất quán đáp án, do gắn thị giác lên một LLM đóng băng |
| V2PE | "Mã hoá vị trí thị giác biến thiên" | Phân bổ mã hoá vị trí học được theo từng phương thức; bản kế nhiệm M-RoPE của InternVL3 |
| ViR | "Router độ phân giải" | Classifier nhỏ chọn độ phân giải tối thiểu cần cho mỗi truy vấn trước khi mã hoá, tiết kiệm token suy luận |
| DvD | "Triển khai tách rời" | Vision encoder trên một GPU, LLM trên GPU khác, bàn giao bằng luồng; nhân đôi thông lượng cho VLM lớn |
| InternVL-U | "Hiểu + sinh hợp nhất" | Bản tiếp nối 2026 thêm head sinh ảnh vào backbone tiền huấn luyện nguyên bản |
| Interleaved corpus | "OBELICS / MMC4" | Tài liệu có văn bản và ảnh theo thứ tự đọc tự nhiên; nguyên liệu thô cho tiền huấn luyện nguyên bản |

## Đọc thêm

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)

---

[← Bài trước: Họ Qwen-VL và Video Dynamic-FPS](../../09-qwen-vl-family-dynamic-fps/docs/vi.md) · [Bài sau: Chameleon và các mô hình đa phương thức chỉ dùng token với early fusion (hợp nhất sớm) →](../../11-chameleon-early-fusion-tokens/docs/vi.md)
