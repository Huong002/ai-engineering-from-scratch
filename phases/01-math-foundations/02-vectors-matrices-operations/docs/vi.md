# Vectơ, Ma trận & Các phép toán

> Mạng nơ-ron nào cũng chỉ là phép nhân ma trận cộng thêm vài bước.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Giai đoạn 1, Bài 01 (Linear Algebra Intuition)
**Time:** ~60 phút

## Mục tiêu học tập

- Xây dựng lớp Matrix với phép toán element-wise, phép nhân ma trận, chuyển vị (transpose), định thức và nghịch đảo
- Phân biệt phép nhân element-wise với phép nhân ma trận và giải thích khi nào dùng mỗi loại
- Cài đặt một tầng mạng nơ-ron dày đặc (`relu(W @ x + b)`) chỉ bằng lớp Matrix tự viết
- Giải thích quy tắc broadcasting và cách cộng bias hoạt động trong các framework mạng nơ-ron

## Vấn đề

Bạn muốn xây dựng một mạng nơ-ron. Bạn đọc code và thấy dòng này:

```
output = activation(weights @ input + bias)
```

Dấu `@` đó là phép nhân ma trận. `weights` là một ma trận (matrix). `input` là một vectơ (vector). Nếu bạn không biết các phép toán đó làm gì, dòng này là ma thuật. Nếu bạn biết, nó chính là toàn bộ lượt lan truyền xuôi của một tầng chỉ với ba phép toán.

Mỗi bức ảnh mô hình của bạn xử lý là một ma trận các giá trị điểm ảnh. Mỗi embedding từ là một vectơ. Mỗi tầng của mọi mạng nơ-ron là một phép biến đổi ma trận. Bạn không thể xây dựng hệ thống AI nếu không thành thạo các phép toán ma trận, cũng như bạn không thể viết code mà không hiểu biến.

Bài học này xây dựng sự thành thạo đó từ đầu.

## Khái niệm

### Vectơ: danh sách số có thứ tự

Một vectơ là danh sách các số kèm hướng và độ lớn. Trong AI, vectơ biểu diễn các điểm dữ liệu, các đặc trưng hoặc các tham số.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

Một vectơ 2 chiều `[3, 4]` trỏ tới tọa độ (3, 4) trên mặt phẳng. Độ dài (độ lớn) của nó là 5 (tam giác 3-4-5).

### Ma trận: lưới các con số

Một ma trận là một lưới 2 chiều. Gồm hàng và cột. Ma trận m x n có m hàng và n cột.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

Trong mạng nơ-ron, ma trận trọng số biến đổi vectơ đầu vào thành vectơ đầu ra. Một tầng có 784 đầu vào và 128 đầu ra dùng ma trận trọng số 128x784.

### Vì sao kích thước (shape) lại quan trọng

Phép nhân ma trận có một quy tắc nghiêm ngặt: `(m x n) @ (n x p) = (m x p)`. Các chiều bên trong phải khớp nhau.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

Nếu bạn gặp lỗi lệch kích thước trong PyTorch, đây chính là lý do.

### Bản đồ các phép toán

| Phép toán | Nó làm gì | Ứng dụng trong mạng nơ-ron |
|-----------|-------------|-------------------|
| Cộng | Kết hợp element-wise | Cộng bias vào đầu ra |
| Nhân vô hướng (scalar) | Nhân tỉ lệ mọi phần tử | Learning rate * gradient |
| Nhân ma trận | Biến đổi vectơ | Lượt lan truyền xuôi của tầng |
| Chuyển vị | Hoán đổi hàng và cột | Lan truyền ngược |
| Định thức | Tóm tắt bằng một con số | Kiểm tra khả năng nghịch đảo |
| Nghịch đảo | Hoàn tác một phép biến đổi | Giải hệ phương trình tuyến tính |
| Ma trận đơn vị | Ma trận không làm gì cả | Khởi tạo, kết nối residual |

### Element-wise và phép nhân ma trận

Sự phân biệt này khiến người mới học liên tục nhầm lẫn.

Element-wise: nhân các vị trí tương ứng với nhau. Cả hai ma trận phải có cùng kích thước.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Phép nhân ma trận: tích vô hướng (dot product) của hàng và cột. Các chiều bên trong phải khớp nhau.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Hai phép toán khác nhau, kết quả khác nhau, quy tắc khác nhau.

### Broadcasting

Khi bạn cộng một vectơ bias vào một ma trận đầu ra, các kích thước không khớp. Broadcasting kéo giãn mảng nhỏ hơn cho vừa.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Mọi framework hiện đại đều tự động làm việc này. Hiểu nó giúp bạn không bối rối khi kích thước có vẻ sai nhưng code vẫn chạy.

```figure
vector-projection
```

## Tự tay làm

### Bước 1: Lớp Vector

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Bước 2: Lớp Matrix với các phép toán cốt lõi

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Bước 3: Xem nó hoạt động

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Bước 4: Kết nối với mạng nơ-ron

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

Đây là một lớp dense đơn: `output = relu(W @ x + b)`. Mọi lớp dense trong mọi mạng nơ-ron đều làm đúng như vậy.

## Áp dụng

NumPy làm mọi thứ ở trên chỉ với ít dòng hơn và nhanh hơn nhiều bậc độ lớn.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Toán tử `@` trong Python gọi `__matmul__`. NumPy triển khai nó bằng các routine BLAS tối ưu viết bằng C và Fortran. Cùng một phép toán, nhanh hơn 100x.

Broadcasting trong NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy tự động broadcast bias 1D trên cả hai hàng. Đây là cách phép cộng bias hoạt động trong mọi framework mạng nơ-ron.

## Đóng gói

Bài học này tạo ra một prompt để dạy các phép toán ma trận thông qua trực giác hình học. Xem `outputs/prompt-matrix-operations.md`.

Lớp Matrix xây dựng ở đây là nền tảng cho framework mạng nơ-ron mini mà chúng ta xây ở Giai đoạn 3, Bài 10.

## Bài tập

1. **Xác minh ma trận nghịch đảo.** Nhân `A @ A.inverse_2x2()` và xác nhận bạn nhận được ma trận đơn vị. Thử với ba ma trận 2x2 khác nhau. Điều gì xảy ra khi định thức bằng không?

2. **Cài đặt nghịch đảo 3x3.** Mở rộng lớp Matrix để tính nghịch đảo cho ma trận 3x3 bằng phương pháp adjugate. Kiểm tra với `np.linalg.inv` của NumPy.

3. **Xây một mạng hai lớp.** Chỉ dùng lớp Matrix của bạn (không dùng NumPy), tạo một mạng nơ-ron hai lớp: đầu vào (3) -> ẩn (4) -> đầu ra (2). Khởi tạo trọng số ngẫu nhiên, chạy một lượt lan truyền xuôi, và xác minh mọi kích thước đều đúng.

## Thuật ngữ chính

| Thuật ngữ | Người ta nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Vector (vectơ) | "Một mũi tên" | Một danh sách số có thứ tự. Trong AI: một điểm trong không gian nhiều chiều. |
| Matrix (ma trận) | "Một bảng số" | Một phép biến đổi tuyến tính. Nó ánh xạ vectơ từ không gian này sang không gian khác. |
| Matrix multiply | "Chỉ cần nhân các số" | Tích vô hướng giữa từng hàng của ma trận thứ nhất và từng cột của ma trận thứ hai. Thứ tự rất quan trọng. |
| Transpose (chuyển vị) | "Lật nó lại" | Hoán đổi hàng và cột. Biến ma trận m x n thành n x m. Cực kỳ quan trọng trong lan truyền ngược. |
| Determinant | "Một con số nào đó lấy từ ma trận" | Đo mức ma trận phóng đại diện tích (2D) hay thể tích (3D). Bằng không nghĩa là phép biến đổi ép dẹt một chiều. |
| Inverse | "Hoàn tác ma trận" | Ma trận đảo ngược phép biến đổi. Chỉ tồn tại khi định thức khác không. |
| Identity matrix | "Ma trận nhàm chán" | Tương đương ma trận của phép nhân với 1. Dùng trong các kết nối residual (ResNets). |
| Broadcasting | "Sửa kích thước bằng phép thuật" | Kéo giãn một array nhỏ hơn để khớp với array lớn hơn bằng cách lặp lại theo các chiều còn thiếu. |
| Element-wise | "Nhân bình thường" | Nhân các vị trí tương ứng. Cả hai array phải cùng kích thước (hoặc broadcast được). |

## Đọc thêm

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - trực giác hình ảnh cho mọi phép toán được đề cập ở đây
- [Tài liệu NumPy về broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - các quy tắc chính xác mà NumPy tuân theo
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - tài liệu tham khảo ngắn gọn cho đại số tuyến tính chuyên cho ML
