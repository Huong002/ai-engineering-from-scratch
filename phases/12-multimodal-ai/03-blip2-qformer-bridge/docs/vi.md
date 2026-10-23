# Từ CLIP đến BLIP-2 — Q-Former như cầu nối đa phương thức

> CLIP căn chỉnh ảnh và văn bản nhưng không thể sinh caption (chú thích), trả lời câu hỏi hay trò chuyện. BLIP-2 (Salesforce, 2023) giải quyết việc đó bằng một cầu nối nhỏ có thể huấn luyện: 32 query token (token truy vấn) học được chú ý trên đặc trưng của một ViT đã frozen, qua cross-attention, rồi được ghép thẳng vào luồng đầu vào của một LLM đã frozen. 188M tham số của cầu nối đã nối một LLM 11B với ViT-g/14. Mọi VLM dựa trên adapter cho tới 2026 — MiniGPT-4, InstructBLIP, họ hàng của LLaVA — đều là hậu duệ. Bài này đọc kiến trúc của Q-Former, giải thích cách huấn luyện hai giai đoạn của nó, và tự tay dựng một phiên bản đồ chơi đưa visual token vào một bộ giải mã văn bản đã frozen.

**Type:** Build
**Languages:** Python (stdlib, cross-attention + learnable-query demo)
**Prerequisites:** Giai đoạn 12, Bài 02 (CLIP), Giai đoạn 7 (Transformer)
**Time:** ~180 phút

## Mục tiêu học tập

- Giải thích vì sao một bottleneck (nút cổ chai) có thể huấn luyện nằm giữa vision encoder (bộ mã hoá ảnh) đã frozen (đóng băng) và LLM đã frozen lại thắng finetuning đầu-cuối về chi phí và độ ổn định.
- Cài đặt một khối cross-attention (chú ý chéo) trong đó một tập truy vấn học được cố định chú ý vào đặc trưng ảnh bên ngoài.
- Đi qua quá trình tiền huấn luyện hai giai đoạn của BLIP-2: biểu diễn (ITC + ITM + ITG) rồi sinh (mất mát LM với decoder đã frozen).
- So sánh Q-Former với projector MLP đơn giản hơn dùng trong LLaVA và lập luận khi nào mỗi lựa chọn thắng.

## Vấn đề

Bạn có một ViT đã frozen sinh ra 256 patch token chiều 1408 cho mỗi ảnh. Bạn có một LLM 7B đã frozen cần embedding token chiều 4096. Cầu nối hiển nhiên — một lớp tuyến tính từ 1408 sang 4096 — vẫn chạy được, nhưng đưa cả 256 patch token vào ngữ cảnh của LLM tốn thêm 256 token mỗi ảnh. Với batch 32 ảnh thì riêng phương thức thị giác đã tiêu 8192 token.

Câu hỏi của BLIP-2: bạn có thể nén biểu diễn ảnh 256 token xuống ít token hơn nhiều (ví dụ 32) mà vẫn giữ đủ thông tin để LLM tạo caption, trả lời câu hỏi và suy luận về ảnh không? Và bạn có thể huấn luyện cầu nối này mà không chạm vào các backbone đã frozen, giữ chi phí huấn luyện đúng bằng tham số của cầu nối không?

Câu trả lời: Q-Former — một query transformer (bộ biến đổi truy vấn). 32 vector "query" học được cross-attend vào patch token của ViT, tạo ra một bản tóm tắt thị giác 32 token để LLM dùng. Tổng cộng 188M tham số. Được huấn luyện bằng các mục tiêu contrastive, matching và sinh trước khi chạm tới LLM.

## Khái niệm

### Truy vấn học được

Mẹo cốt lõi của Q-Former: thay vì để các token văn bản của LLM chú ý vào patch ảnh, hãy thêm một tập 32 vector truy vấn học được `Q` và để *chính chúng* chú ý vào patch ảnh. Các truy vấn là tham số của mô hình — chúng được học trong lúc huấn luyện và cùng 32 truy vấn đó được dùng cho mọi ảnh.

Sau cross-attention, mỗi truy vấn giữ một bản tóm tắt nén của ảnh — "mô tả vật thể chính", "mô tả nền", "đếm số vật thể", v.v. Các truy vấn không thực sự chuyên trách theo nhãn ngữ nghĩa; chúng học bất kỳ cách mã hoá nào làm mất mát hạ nguồn giảm.

### Kiến trúc

Q-Former là một transformer nhỏ (12 lớp, ~100M tham số) với hai nhánh:

1. Nhánh truy vấn: 32 vector truy vấn đi qua self-attention (giữa chúng với nhau), rồi cross-attention trên patch token của ViT đã frozen, rồi FFN.
2. Nhánh văn bản: một text encoder kiểu BERT chia sẻ trọng số self-attention và FFN với nhánh truy vấn. Cross-attention bị tắt với nhánh văn bản.

Lúc huấn luyện cả hai nhánh cùng chạy. Truy vấn và văn bản tương tác qua self-attention chia sẻ, nghĩa là truy vấn có thể điều kiện theo văn bản với những tác vụ cần điều đó (ITM, ITG). Lúc suy luận để bàn giao cho VLM, chỉ truy vấn chạy qua, cho ra 32 visual token.

### Huấn luyện hai giai đoạn

BLIP-2 tiền huấn luyện trong hai giai đoạn:

Giai đoạn 1: học biểu diễn (không có LLM). Ba mất mát:
- ITC (image-text contrastive): contrastive kiểu CLIP giữa token truy vấn đã pool và token CLS của văn bản.
- ITM (image-text matching): bộ phân loại nhị phân — cặp ảnh-văn bản này có khớp không? Khai thác hard negative.
- ITG (image-grounded text generation): đầu LM nhân quả trên văn bản, điều kiện theo các truy vấn. Buộc truy vấn phải mã hoá nội dung có thể sinh ra văn bản.

Chỉ Q-Former được huấn luyện. ViT đã frozen. Không có LLM tham gia.

Giai đoạn 2: học sinh. Gắn một LLM đã frozen (OPT-2.7B hay Flan-T5-XL, v.v.). Chiếu 32 đầu ra truy vấn sang chiều embedding của LLM bằng một lớp tuyến tính nhỏ. Ghép chúng lên trước prompt văn bản. Chỉ huấn luyện phép chiếu tuyến tính và Q-Former theo mất mát LM trên chuỗi prompt + ảnh + caption đã nối.

Sau giai đoạn 2, Q-Former + phép chiếu chính là toàn bộ adapter thị giác. Lúc suy luận: ảnh → ViT → Q-Former → chiếu tuyến tính → ghép lên trước văn bản → LLM đã frozen phát ra đầu ra.

### Bài toán tham số

BLIP-2 với ViT-g/14 (1.1B, frozen) + OPT-6.7B (6.7B, frozen) + Q-Former (188M, được huấn luyện) = tổng 8B, 188M được huấn luyện. Riêng Q-Former chiếm ~2.4% tham số của cả chồng. Chi phí huấn luyện phản ánh điều đó: vài ngày trên một vài A100 so với vài tuần nếu làm đầu-cuối.

Chất lượng: BLIP-2 ngang hoặc hơn Flamingo-80B trên VQA zero-shot mà nhỏ hơn 50 lần. Cầu nối này hiệu quả.

### InstructBLIP và Q-Former nhận biết chỉ dẫn

InstructBLIP (2023) mở rộng Q-Former với một đầu vào thêm: chính văn bản chỉ dẫn. Lúc cross-attention, các truy vấn giờ truy cập được cả patch ảnh lẫn chỉ dẫn. Truy vấn có thể chuyên biệt theo từng chỉ dẫn ("đếm số xe", "mô tả tâm trạng") thay vì học một bản tóm tắt cố định duy nhất. Điểm benchmark tăng trên các tác vụ held-out.

### MiniGPT-4 và hướng chỉ dùng projector

MiniGPT-4 giữ Q-Former nhưng chỉ huấn luyện phép chiếu tuyến tính đầu ra trong khi đóng băng mọi thứ còn lại. Rẻ, nhưng cái giá là chất lượng — truy vấn là của BLIP-2, không phải của bạn. Tốt cho lặp nhanh, không phải kiến trúc tốt nhất.

### Vì sao LLaVA chọn cách đơn giản hơn

LLaVA (2023, Bài 12.05) thay Q-Former bằng một MLP 2 lớp thuần chiếu mọi patch token của ViT vào không gian của LLM — 576 token mỗi ảnh với lưới 24x24, tất cả đưa vào LLM. Nén kém hơn nhưng cho LLM chú ý trên patch thô. Lúc đó điều này gây tranh cãi; tới cuối 2023 nó đã chiếm ưu thế vì dữ liệu chỉ dẫn thị giác (LLaVA-Instruct-150k) chứng minh MLP có thể được huấn luyện để giữ đủ tín hiệu. Đánh đổi: ngữ cảnh của LLaVA đầy nhanh hơn, nhưng nó mở rộng tự nhiên sang nhiều ảnh và video.

Tới 2026 lĩnh vực tách đôi: Q-Former sống ở nơi ngân sách token quan trọng (video dài, nhiều ảnh); projector MLP chiếm ưu thế ở nơi ưu tiên chất lượng trên mỗi token.

### Gated cross-attention: Flamingo, tổ tiên

Flamingo (Bài 12.04) có trước BLIP-2 và dùng cùng ý tưởng cross-attention nhưng ở mọi lớp của LLM đã frozen, chứ không phải một cầu nối duy nhất. BLIP-2 cho thấy bạn có thể nén về chỉ lớp đầu vào mà vẫn chạy được. Gemini và Idefics kết hợp cả hai: token đầu vào xen kẽ cộng cross-attention có cổng tuỳ chọn cho few-shot trong ngữ cảnh.

### Hậu duệ năm 2026

- Q-Former: BLIP-2, InstructBLIP, MiniGPT-4, và hầu hết mô hình ngôn ngữ-video vì lý do ngân sách token.
- Perceiver resampler: biến thể của Flamingo (Bài 12.04); họ Idefics, Eagle, OmniMAE.
- Projector MLP: LLaVA, LLaVA-NeXT, LLaVA-OneVision, Cambrian-1.
- Attention pool: VILA, PaliGemma.

Cả bốn đều hợp lệ. Câu hỏi quyết định là bạn bị ràng buộc về ngân sách token hay về chất lượng trên mỗi token.

```figure
modality-projection
```

## Áp dụng

`code/main.py` dựng một cross-attention kiểu Q-Former bằng stdlib:

1. Mô phỏng 256 patch token của ảnh (chiều 128).
2. Khởi tạo 32 truy vấn học được (chiều 128).
3. Chạy cross-attention tích vô hướng có scale (Q từ truy vấn, K/V từ patch).
4. Chiếu sang chiều LLM (512) qua một lớp tuyến tính.
5. Xuất ra 32 visual token sẵn sàng cho LLM.

Toàn bộ toán bằng Python thuần (vòng lặp lồng trên vector). Đồ chơi nhưng đúng hình dạng. Ma trận trọng số chú ý được in ra để bạn thấy mỗi truy vấn lấy từ patch nào.

## Đóng gói

Bài này tạo ra `outputs/skill-modality-bridge-picker.md`. Cho một cấu hình VLM mục tiêu (số token của vision encoder, ngân sách ngữ cảnh của LLM, ràng buộc triển khai, mục tiêu chất lượng), nó đề xuất Q-Former vs MLP vs Perceiver resampler kèm lý do ngắn và ước lượng số tham số cho từng cầu nối.

## Bài tập

1. Cài đặt khối cross-attention bằng PyTorch. Kiểm chứng rằng với 32 truy vấn và 256 key/value, ma trận trọng số chú ý là 32 x 256 và mỗi hàng cộng lại bằng 1 sau softmax.

2. Trong giai đoạn 1 của BLIP-2, Q-Former chạy đồng thời ba mất mát: ITC, ITM, ITG. Viết chữ ký forward cho mỗi cái bằng pseudo-code. Cái nào cần nhánh text encoder hoạt động?

3. So sánh số tham số: Q-Former (12 lớp, ẩn 768) với một projector MLP 2 lớp (1408 → 4096, hai lớp). Ở quy mô LLM nào thì chi phí 188M của Q-Former hoàn vốn về hiệu quả huấn luyện?

4. Đọc Mục 3.2 của bài báo BLIP-2 (arXiv:2301.12597) về cách khởi tạo Q-Former. Giải thích vì sao khởi tạo từ BERT-base (không phải ngẫu nhiên) giúp hội tụ nhanh hơn.

5. Với một video 10 phút ở 1 FPS lấy mẫu thành 60 khung hình, tính chi phí token mỗi khung ở (Q-Former → 32 token/khung) vs (projector MLP → 576 token/khung). Cách nào vừa ngữ cảnh LLM 128k token?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Q-Former | "Querying transformer" | Transformer nhỏ với 32 vector truy vấn học được cross-attend vào đặc trưng ViT đã frozen |
| Learnable queries | "Soft prompt cho thị giác" | Một tập tham số cố định đóng vai trò phía truy vấn của cross-attention; học theo mô hình, dùng chung cho mọi đầu vào |
| Cross-attention | "Q từ đây, K/V từ kia" | Chú ý mà query, key và value đến từ các nguồn khác nhau; cách các truy vấn lấy từ patch của ViT |
| ITC | "Image-text contrastive" | Mất mát kiểu CLIP áp lên truy vấn đã pool của Q-Former so với CLS văn bản |
| ITM | "Image-text matching" | Bộ phân loại nhị phân trên các cặp hard-negative-mined; buộc truy vấn phân biệt các lệch khớp tinh vi |
| ITG | "Image-grounded text generation" | Mất mát LM nhân quả trong đó văn bản được sinh điều kiện theo truy vấn; buộc truy vấn mã hoá nội dung giải mã được thành văn bản |
| Two-stage pretraining | "Biểu diễn rồi sinh" | Giai đoạn 1 huấn luyện riêng Q-Former (ITC/ITM/ITG); giai đoạn 2 gắn LLM đã frozen và chỉ huấn luyện phép chiếu + Q-Former |
| Frozen backbone | "Đừng finetune" | Trọng số vision encoder và LLM cố định; chỉ cầu nối được huấn luyện |
| Projection head | "Tuyến tính sang chiều LLM" | Lớp tuyến tính cuối ánh xạ đầu ra Q-Former sang chiều embedding của LLM |
| Perceiver resampler | "Bản của Flamingo" | Cross-attention với truy vấn học được tương tự, nhưng Flamingo dùng ở mọi lớp thay vì một cầu nối duy nhất |

## Đọc thêm

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — bài báo cốt lõi.
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — tiền nhiệm với bộ ba ITC/ITM/ITG.
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "align before fuse" — tổ tiên khái niệm của huấn luyện giai đoạn 1.
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — Q-Former nhận biết chỉ dẫn.
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — hướng chỉ dùng projector.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — kiến trúc tổng quát cho cross-attention với truy vấn học được.

---

[← Bài trước: CLIP và tiền huấn luyện tương phản thị giác-ngôn ngữ](../../02-clip-contrastive-pretraining/docs/vi.md) · [Bài sau: Flamingo và gated cross-attention (cross-attention có cổng) cho VLM few-shot →](../../04-flamingo-gated-cross-attention/docs/vi.md)
