# Vision Transformers (ViT)

> Ảnh là một lưới các mảnh ảnh. Câu văn là một lưới các token. Cùng một transformer ăn được cả hai.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 7, Bài 05 (Transformer đầy đủ), Giai đoạn 4, Bài 03 (CNN), Giai đoạn 4, Bài 14 (Giới thiệu Vision Transformer)
**Time:** ~45 phút

## Vấn đề

Trước 2020, thị giác máy tính đồng nghĩa với tích chập. Mọi kết quả SOTA trên ImageNet, COCO và các benchmark phát hiện đều dùng backbone CNN. Transformer là để dành cho ngôn ngữ.

Dosovitskiy và cộng sự (2020) — "An Image is Worth 16x16 Words" — chỉ ra rằng bạn có thể bỏ hoàn toàn tích chập. Cắt ảnh thành các patch (mảnh ảnh) kích thước cố định, chiếu tuyến tính từng mảnh thành một embedding, đưa chuỗi đó vào một transformer encoder nguyên bản. Với quy mô đủ lớn (pretrain trên ImageNet-21k hoặc lớn hơn), ViT ngang hoặc vượt các mô hình dựa trên ResNet.

ViT mở đầu cho một mẫu hình rộng hơn vào 2026: một kiến trúc, nhiều phương thức dữ liệu. Whisper tokenize âm thanh. ViT tokenize hình ảnh. Action token cho robot. Pixel token cho video. Transformer không bận tâm — đưa cho nó một chuỗi là nó học.

Đến 2026, ViT và các hậu duệ (DeiT, Swin, DINOv2, ViT-22B, SAM 3) chiếm phần lớn thị giác. CNN vẫn thắng trên thiết bị biên và các tác vụ nhạy cảm độ trễ. Mọi thứ còn lại đều có một ViT đâu đó trong stack.

## Khái niệm

![Ảnh → mảnh ảnh → token → transformer](../assets/vit.svg)

### Bước 1 — patchify

Tách một ảnh `H × W × C` thành chuỗi `N × (P·P·C)` các patch phẳng. Cấu hình điển hình: ảnh `224 × 224`, patch `16 × 16` → 196 patch, mỗi patch 768 giá trị.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Kích thước patch là đòn bẩy. Patch nhỏ hơn = nhiều token hơn, image resolution (độ phân giải ảnh) tốt hơn, chi phí attention bậc hai. Patch lớn hơn = thô hơn, rẻ hơn.

### Bước 2 — embedding tuyến tính

Một ma trận học được duy nhất chiếu từng patch phẳng tới `d_model`. Tương đương một phép tích chập kernel `P`, stride `P`. Trong PyTorch, đây đúng nghĩa là `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` — implementation 2 dòng.

### Bước 3 — gắn token `[CLS]` lên đầu, cộng embedding vị trí

- Gắn lên đầu một `[CLS]` token học được — class token (token lớp). Hidden state cuối của nó là biểu diễn ảnh dùng để phân loại.
- Cộng embedding vị trí học được (bản ViT gốc) hoặc sinusoidal 2D (các biến thể sau).
- Từ 2024+, RoPE mở rộng sang 2D cho vị trí, đôi khi không cần embedding tường minh.

### Bước 4 — transformer encoder chuẩn

Xếp L khối `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`. Giống hệt BERT. Không có tầng nào riêng cho thị giác. Đây chính là điểm nhấn sư phạm của bài báo.

### Bước 5 — head

Với phân loại: lấy hidden state của `[CLS]` → linear → softmax. Với DINOv2 hoặc SAM, bỏ `[CLS]`, dùng thẳng patch embedding (embedding mảnh ảnh).

### Những biến thể đáng kể

| Mô hình | Năm | Thay đổi |
|-------|------|--------|
| ViT | 2020 | Bản gốc. Patch size cố định, global attention đầy đủ. |
| DeiT | 2021 | Distillation; huấn luyện chỉ trên ImageNet-1k. |
| Swin | 2021 | Phân cấp với shifted window. Chi phí dưới bậc hai cố định. |
| DINOv2 | 2023 | Tự giám sát (không nhãn). Đặc trưng thị giác tổng quát tốt nhất. |
| ViT-22B | 2023 | 22B tham số; scaling law vẫn đúng. |
| SigLIP | 2023 | ViT + cặp ngôn ngữ, loss contrastive sigmoid. |
| SAM 3 | 2025 | Segment anything; ViT-Large + promptable mask decoder. |

### Vì sao phải mất một thời gian

ViT cần *rất nhiều* dữ liệu để ngang CNN vì nó không có inductive bias (thiên lệch quy nạp) nào của CNN (bất biến dịch, tính cục bộ). Không có hơn 100M ảnh gán nhãn hoặc pretrain tự giám sát mạnh, CNN vẫn thắng ở cùng mức compute. DeiT sửa được điều này năm 2021 bằng các thủ thuật distillation; DINOv2 sửa hẳn vào 2023 bằng tự giám sát.

```figure
n5-patch-stream
```

## Tự tay làm

Xem `code/main.py`. Patchify + linear embedding thuần stdlib + kiểm tra nhanh. Không huấn luyện — ViT ở quy mô thực tế nào cũng cần PyTorch và nhiều giờ GPU.

### Bước 1: ảnh giả

Một ảnh RGB 24 × 24 dưới dạng danh sách các hàng của tuple `(R, G, B)`. Ta dùng patch 6×6 → 16 patch, mỗi vector embedding 108 chiều.

### Bước 2: patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Thứ tự raster: duyệt theo hàng trên lưới. Mọi ViT đều dùng thứ tự này.

### Bước 3: nhúng tuyến tính

Nhân từng patch phẳng với một ma trận ngẫu nhiên `(patch_flat_size, d_model)`. Kiểm tra shape đầu ra là `(N_patches + 1, d_model)` sau khi gắn `[CLS]` lên đầu.

### Bước 4: đếm tham số cho một ViT thực tế

In số tham số của ViT-Base: 12 layer, 12 head, d=768, patch=16. So với ResNet-50 (~25M). ViT-Base vào khoảng ~86M. ViT-Large ~307M. ViT-Huge ~632M.

## Áp dụng

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**Embedding DINOv2 là mặc định năm 2026 cho đặc trưng ảnh.** Đóng băng backbone, huấn luyện một head nhỏ. Dùng được cho phân loại, truy hồi, phát hiện, sinh mô tả. Checkpoint DINOv2 của Meta vượt CLIP trên mọi tác vụ thị giác không liên quan văn bản.

**Chọn kích thước patch.** Mô hình nhỏ dùng 16×16 (ViT-B/16). Dự đoán dày đặc (segmentation) dùng 8×8 hoặc 14×14 (SAM, DINOv2). Mô hình rất lớn dùng 14×14.

## Đóng gói

Xem `outputs/skill-vit-configurator.md`. Skill này chọn biến thể ViT và kích thước patch cho một tác vụ thị giác mới, dựa trên kích thước dataset, độ phân giải và ngân sách compute.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Kiểm tra số patch bằng `(H/P) * (W/P)` và chiều patch phẳng bằng `P*P*C`.
2. **Trung bình.** Cài embedding vị trí sinusoidal 2D — hai mã sinusoidal độc lập cho `row` và `col` của mỗi patch, ghép lại. Đưa chúng vào một ViT PyTorch nhỏ và so độ chính xác với embedding vị trí học được trên CIFAR-10.
3. **Khó.** Dựng một ViT 3 lớp (PyTorch), huấn luyện trên 1.000 ảnh MNIST với patch 4×4. Đo độ chính xác test. Giờ thêm pretrain kiểu DINOv2 trên chính 1.000 ảnh đó (đơn giản hoá: chỉ huấn luyện encoder dự đoán patch embedding từ các patch bị che). Độ chính xác có tăng không?

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Patch | "Token của vision transformer" | Vector phẳng các giá trị pixel của một vùng `P × P × C` trong ảnh. |
| Patchify | "Cắt + làm phẳng" | Cắt ảnh thành các patch không chồng lấn, làm phẳng mỗi patch thành một vector. |
| `[CLS]` token | "Bản tóm tắt ảnh" | Token học được gắn lên đầu; embedding cuối của nó là biểu diễn ảnh. |
| Inductive bias | "Thứ mô hình giả định sẵn" | ViT có ít prior hơn CNN; cần nhiều dữ liệu hơn để bù khoảng cách. |
| DINOv2 | "ViT tự giám sát" | Huấn luyện không nhãn bằng tăng cường ảnh + teacher momentum. Đặc trưng ảnh tổng quát tốt nhất năm 2026. |
| SigLIP | "Kế nhiệm CLIP" | ViT + text encoder huấn luyện bằng loss contrastive sigmoid; tốt hơn CLIP ở cùng compute. |
| Swin | "ViT dạng cửa sổ" | ViT phân cấp với attention cục bộ + shifted window; dưới bậc hai. |
| Register tokens | "Chiêu 2023" | Vài token học được thêm vào để hút các attention sink; cải thiện đặc trưng DINOv2. |

## Đọc thêm

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — bài báo ViT.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — bản vá register-token cho DINOv2.

---

[← Bài trước: T5, BART](../../08-t5-bart-encoder-decoder/docs/vi.md) · [Bài sau: Audio Transformer (transformer cho âm thanh) →](../../10-audio-transformers-whisper/docs/vi.md)
