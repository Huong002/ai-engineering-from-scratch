# Show-o và các mô hình hợp nhất khuếch tán rời rạc

> Transfusion trộn lẫn biểu diễn liên tục và rời rạc. Show-o (Xie et al., tháng 8/2024) đi theo hướng ngược lại: text token (token văn bản) dùng dự đoán token kế tiếp nhân quả, token ảnh dùng masked discrete diffusion theo tinh thần MaskGIT. Cả hai nằm trong một transformer duy nhất với mặt nạ chú ý lai. Kết quả là một unified model (mô hình hợp nhất) gộp VQA, text-to-image, inpainting và sinh đa phương thức hỗn hợp trên một backbone, một tokenizer cho mỗi phương thức, một công thức loss duy nhất (next-token mở rộng thành dự đoán có che). Bài này đi qua thiết kế Show-o — vì sao masked discrete diffusion là bộ sinh ảnh song song, ít bước — và so sánh với Transfusion và Emu3.

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Giai đoạn 12, Bài 13 (Transfusion)
**Time:** ~120 phút

## Mục tiêu học tập

- Giải thích masked discrete diffusion: sampling schedule (lịch lấy mẫu) che token đồng đều rồi yêu cầu transformer khôi phục chúng.
- So sánh giải mã ảnh song song (Show-o, MaskGIT) với giải mã ảnh tự hồi quy (Chameleon, Emu3) về tốc độ và chất lượng.
- Kể tên ba tác vụ Show-o xử lý trong một checkpoint: T2I, VQA, inpainting ảnh.
- Chọn một lịch che (cosine, tuyến tính, cắt ngắn) và lý giải ảnh hưởng của nó tới chất lượng mẫu.

## Vấn đề

Huấn luyện hai loss của Transfusion chạy được nhưng động học phức tạp hơn — loss khuếch tán liên tục nằm ở thang số khác với loss NTP rời rạc. Cân bằng trọng số loss là một bài toán tìm kiếm siêu tham số. Kiến trúc hiệu quả nhưng phức tạp.

Câu trả lời của Show-o: giữ cả hai phương thức ở dạng rời rạc (như Chameleon), nhưng sinh ảnh song song bằng discrete diffusion (khuếch tán rời rạc) có che thay vì tuần tự. Mục tiêu huấn luyện trở thành một phép dự đoán masked token (token bị che) duy nhất, tổng quát hoá dự đoán token kế tiếp một cách tự nhiên.

## Khái niệm

### Khuếch tán rời rạc có che (MaskGIT)

Thủ thuật MaskGIT của Chang et al. (2022) rất thanh thoát. Bắt đầu từ một ảnh bị che hoàn toàn: mọi token đều là id `<MASK>` đặc biệt, tức absorbing state (trạng thái hấp thụ). Ở mỗi bước, dự đoán song song tất cả token bị che, rồi giữ lại K dự đoán tự tin nhất và che lại phần còn lại. Sau khoảng 8-16 vòng lặp, mọi token được điền đầy. Lịch quyết định bao nhiêu token được mở che mỗi bước là thứ được tinh chỉnh — lịch cosine chạy tốt.

Huấn luyện rất đơn giản: lấy mẫu một tỉ lệ che đồng đều trong [0, 1], áp lên các token VQ của ảnh, huấn luyện transformer khôi phục những token bị che. Đúng những gì BERT làm với văn bản, nhân rộng lên cho sinh ảnh.

### Show-o: một transformer, mặt nạ lai

Show-o đặt MaskGIT vào bên trong một transformer ngôn ngữ nhân quả. Mặt nạ chú ý như sau:

- Token văn bản: nhân quả (LLM tiêu chuẩn).
- Token ảnh: song hướng đầy đủ trong khối ảnh (để các token bị che có thể thấy mọi token ảnh khác trong lúc dự đoán).
- Text-to-image: văn bản chú ý tới ảnh phía trước, ảnh chú ý tới văn bản phía trước.

Huấn luyện luân phiên giữa:
1. NTP tiêu chuẩn trên chuỗi văn bản.
2. Mẫu T2I: văn bản → ảnh với token ảnh bị che, loss dự đoán token bị che.
3. Mẫu VQA: ảnh → văn bản với token văn bản bị che (thực ra chỉ là NTP).

Loss hợp nhất là cross-entropy trên các token `<MASK>`, bao trùm cả NTP văn bản (chỉ token cuối là "bị che") và khuếch tán có che cho ảnh (một tập con ngẫu nhiên bị che).

### Lấy mẫu song song

Show-o sinh một ảnh trong khoảng 16 bước thay vì ~1000 (tự hồi quy từng token) hay ~20 (khuếch tán). Ở mỗi bước, dự đoán song song mọi token bị che; chốt K dự đoán tự tin nhất; lặp lại.

So sánh:
- Chameleon / Emu3 (tự hồi quy trên token): N_tokens lượt forward, thường 1024-4096 mỗi ảnh.
- Transfusion (khuếch tán liên tục): ~20 bước, mỗi bước một lượt transformer đầy đủ.
- Show-o (khuếch tán rời rạc có che): ~16 bước, mỗi bước một lượt transformer đầy đủ.

Show-o nhanh hơn Chameleon ở các mô hình cùng cỡ, gần bằng số bước của Transfusion với chi phí mỗi bước thấp hơn (logits trên từ vựng rời rạc so với loss MSE liên tục).

### Các tác vụ trong một checkpoint

Show-o hỗ trợ bốn tác vụ khi suy luận, chọn theo định dạng prompt:

- Sinh văn bản: đầu ra văn bản tự hồi quy tiêu chuẩn.
- VQA: ảnh vào, văn bản ra.
- T2I: văn bản vào, ảnh ra qua khuếch tán rời rạc có che.
- Inpainting: ảnh với một số token bị che, điền vào.

Khả năng inpainting có sẵn miễn phí nhờ huấn luyện dự đoán có che. Che một vùng của lưới token VQ, đưa phần còn lại cùng một prompt văn bản, dự đoán các token bị che.

### Lịch che

Lịch quyết định bao nhiêu token được mở che mỗi bước định hình chất lượng. Show-o khuyến nghị cosine:

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

Ở bước 0, mọi token bị che (tỉ lệ 1.0). Ở bước T, không token nào bị che. Cosine dồn khối lượng vào các tỉ lệ tầm trung, nơi dự đoán mang nhiều thông tin nhất. Lịch tuyến tính cũng chạy được nhưng chững lại nhanh hơn.

### Show-o2

Show-o2 (bản tiếp theo năm 2025, arXiv 2506.15564) mở rộng Show-o: nền LLM lớn hơn, tokenizer tốt hơn, lịch che cải thiện. Vẫn cùng khuôn kiến trúc.

### Show-o nằm ở đâu

Trong hệ phân loại của 2026:

- Token rời rạc + NTP: Chameleon, Emu3. Đơn giản nhưng suy luận chậm.
- Token rời rạc + khuếch tán có che: Show-o, MaskGIT, LlamaGen, Muse. Lấy mẫu song song, vẫn mất mát do tokenizer.
- Liên tục + khuếch tán: Transfusion, MMDiT, DiT. Chất lượng cao nhất, huấn luyện phức tạp hơn.
- Liên tục + flow matching trong một VLM: JanusFlow, InternVL-U. Mới nhất.

Chọn theo tác vụ: Show-o khi bạn muốn T2I + inpainting + VQA trong một mô hình mở với tốc độ hợp lý; Transfusion khi chất lượng là trên hết và bạn đủ nguồn lực cho bộ máy hai loss.

```figure
masked-diffusion-unmask
```

## Áp dụng

`code/main.py` mô phỏng lấy mẫu của Show-o:

- Một lưới đồ chơi gồm 16 token VQ.
- Một "transformer" giả lập dự đoán logits dựa trên prompt và các token hiện chưa bị che.
- Lấy mẫu có che song song qua 8 bước với lịch cosine.
- In ra các trạng thái trung gian (diễn biến mẫu mặt nạ) và các token cuối cùng.

Chạy thử, xem mặt nạ tan dần từng bước.

## Đóng gói

Bài này tạo ra `outputs/skill-unified-gen-model-picker.md`. Với một sản phẩm cần cả hiểu (VQA, captioning) lẫn sinh (T2I, inpainting) trong ràng buộc trọng số mở, bài giúp chọn giữa họ Show-o, họ Transfusion/MMDiT và họ Emu3 / Chameleon kèm các đánh đổi cụ thể.

## Bài tập

1. Khuếch tán rời rạc có che lấy mẫu trong ~16 bước. Sao không phải 1? Điều gì hỏng nếu bạn mở che mọi token ngay ở bước 0?

2. Inpainting miễn phí với khuếch tán có che. Đề xuất một ca sử dụng sản phẩm (thật hoặc giả định) nơi inpainting của Show-o thắng một mô hình chuyên dụng.

3. Lịch cosine so với lịch tuyến tính: vẽ số token được mở che mỗi bước với T=8. Cái nào cân bằng hơn?

4. Một ảnh 512x512 của Show-o là 1024 token. Với từ vựng K=16384, mô hình phát ra 1024 * log2(16384) = 14,336 bit (~1.75 KiB) dữ liệu. Stable Diffusion xuất ra 512*512*24 bit = 6,291,456 bit (~768 KiB) điểm ảnh thô. Tỉ lệ nén là bao nhiêu và nó đổi lấy chất lượng gì?

5. Đọc LlamaGen (arXiv:2406.06525). Mô hình ảnh tự hồi quy theo lớp của LlamaGen khác cách tiếp cận có che của Show-o ở đâu?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Masked discrete diffusion | "MaskGIT-style" | Huấn luyện dự đoán token bị che; khi suy luận, lặp lại việc mở che các dự đoán tự tin nhất |
| Cosine schedule | "Unmask schedule" | Độ giảm của tỉ lệ che qua các bước suy luận; dồn mức tăng độ tự tin vào vùng tầm trung |
| Parallel decoding | "All tokens at once" | Mỗi bước dự đoán toàn bộ chuỗi token bị che trong một lượt forward, rồi chốt top-K |
| Hybrid attention | "Causal + bidirectional" | Mặt nạ nhân quả trên token văn bản và song hướng trong các khối ảnh |
| Inpainting | "Fill-in generation" | Điều kiện hoá trên một ảnh có vài token bị che, dự đoán những token thiếu; miễn phí nhờ mục tiêu huấn luyện |
| Commitment rate | "Top-K per step" | Số token được tuyên bố "xong" mỗi vòng lặp; điều khiển đánh đổi giữa suy luận và chất lượng |

## Đọc thêm

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)

---

[← Bài trước: Transfusion: Văn bản autoregressive (tự hồi quy) + Ảnh diffusion (khuếch tán) trong một Transformer](../../13-transfusion-autoregressive-diffusion/docs/vi.md) · [Bài sau: Janus-Pro: Bộ mã hoá tách rời cho mô hình đa phương thức hợp nhất →](../../15-janus-pro-decoupled-encoders/docs/vi.md)
