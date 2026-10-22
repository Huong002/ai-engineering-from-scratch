# Thiết lập GPU & Cloud

> Huấn luyện trên CPU thì ổn cho việc học. Huấn luyện cho thật thì cần GPU.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 0, Bài 01
**Time:** ~45 phút

## Mục tiêu học tập

- Kiểm tra GPU cục bộ có sẵn hay không bằng `nvidia-smi` và API CUDA của PyTorch
- Cấu hình Google Colab với GPU T4 để thử nghiệm miễn phí trên cloud
- Đo hiệu năng nhân ma trận trên CPU so với GPU và tính mức tăng tốc
- Ước lượng mô hình lớn nhất vừa với VRAM của bạn bằng quy tắc kinh nghiệm fp16

## Vấn đề

Hầu hết bài học trong các phase 1-3 chạy tốt trên CPU. Nhưng khi bạn bắt đầu huấn luyện CNN, transformer hoặc LLM (phase 4 trở đi), bạn cần GPU tăng tốc. Một lần huấn luyện mất 8 giờ trên CPU chỉ mất 10 phút trên GPU.

Bạn có ba lựa chọn: GPU cục bộ, GPU trên cloud, hoặc Google Colab (miễn phí).

## Khái niệm

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## Tự tay làm

### Lựa chọn 1: GPU NVIDIA cục bộ

Kiểm tra xem bạn có GPU hay không:

```bash
nvidia-smi
```

Cài PyTorch với CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Lựa chọn 2: Google Colab

1. Truy cập [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. Chạy `!nvidia-smi` để kiểm tra

Tải notebook (sổ ghi chú) từ khóa học này trực tiếp lên Colab.

### Lựa chọn 3: GPU trên cloud

Với Lambda Labs, RunPod hoặc Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Không có GPU? Không sao.

Hầu hết bài học đều chạy được trên CPU. Những bài cần GPU sẽ nói rõ và kèm liên kết Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Tự tay làm: đo hiệu năng GPU so với CPU

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Bài tập

1. Chạy benchmark trên và so sánh thời gian CPU với GPU
2. Nếu bạn không có GPU, hãy chạy nó trên Google Colab và so sánh
3. Kiểm tra bạn có bao nhiêu bộ nhớ GPU và ước lượng mô hình lớn nhất bạn có thể chứa (quy tắc kinh nghiệm: 2 byte cho mỗi tham số với fp16)

## Thuật ngữ chính

| Thuật ngữ | Người ta thường nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| CUDA | "GPU programming" | Nền tảng tính toán song song của NVIDIA cho phép bạn chạy mã trên GPU |
| VRAM | "GPU memory" | RAM đồ họa trên GPU, tách biệt với RAM hệ thống. Giới hạn kích thước mô hình. |
| fp16 | "Half precision" | Số dấu phẩy động 16-bit, dùng một nửa bộ nhớ so với fp32 mà mất rất ít độ chính xác |
| Tensor Core | "Fast matrix hardware" | Lõi GPU chuyên dụng cho phép nhân ma trận, nhanh hơn 4-8x so với lõi thường |

---

[← Bài trước: Git & Collaboration](../../02-git-and-collaboration/docs/vi.md) · [Bài sau: API & Key →](../../04-apis-and-keys/docs/vi.md)
