# Từ điển thuật ngữ Phase 00 — định nghĩa Wikipedia và code minh họa

Các định nghĩa dưới đây được **diễn đạt/dịch sang tiếng Việt từ trang Wikipedia được liên kết**, không chép nguyên văn. “Vai trò trong ML” là phần giải thích ứng dụng, tách riêng khỏi định nghĩa toán học. Các snippet Python do tài liệu này viết để minh họa, không lấy từ Wikipedia.

Trong bộ Anki, mỗi note giữ Front, Back, Group, Topic và Tags; mặt sau chia thành Định nghĩa/Công thức, Vai trò/Các thành phần, Ví dụ, Code và Nguồn.

## Thuật ngữ

### Gradient (grad) là gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Gradient và tối ưu hóa

**Định nghĩa (Wikipedia):** Với hàm khả vi trả về một số, gradient là vector gồm các đạo hàm riêng theo từng biến. Tại một điểm, nó chỉ hướng hàm tăng nhanh nhất; độ lớn của gradient là tốc độ tăng theo hướng đó.

**Vai trò trong ML:** Trong ML, gradient của loss theo các tham số cho biết loss nhạy thế nào với từng tham số. Muốn giảm loss, thuật toán thường cập nhật tham số ngược hướng gradient.

**Ví dụ:** Với f(x,y)=x²+y², gradient là [2x,2y]. Tại (4,3), gradient [8,6] chỉ hướng tăng; đi ngược hướng đó tiến về điểm thấp nhất (0,0).

**Code minh họa:**

```python
x, y = 4.0, 3.0
grad = [2*x, 2*y]
print(grad)  # [8.0, 6.0]
```

**Nguồn:** [Wikipedia: Gradient](https://en.wikipedia.org/wiki/Gradient)

### Loss / hàm mất mát là gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Loss và tối ưu hóa

**Định nghĩa (Wikipedia):** Hàm mất mát ánh xạ một sự kiện hoặc các giá trị đầu vào thành một số thực biểu thị chi phí. Trong ML, nó lượng hóa mức không phù hợp giữa dự đoán và mục tiêu theo một tiêu chí cụ thể.

**Vai trò trong ML:** Quá trình huấn luyện tìm tham số làm loss nhỏ hơn. Loss không phải một sai số duy nhất bắt buộc: kết quả phụ thuộc vào hàm mất mát được chọn.

**Ví dụ:** Với loss bình phương, đáp án 5 và dự đoán 3 cho loss (3−5)²=4.

**Code minh họa:**

```python
target, pred = 5.0, 3.0
loss = (pred - target) ** 2
print(loss)  # 4.0
```

**Nguồn:** [Wikipedia: Loss function](https://en.wikipedia.org/wiki/Loss_function); [Wikipedia: Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)

### Weight (trọng số w) có vai trò gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Tham số mô hình

**Định nghĩa (Wikipedia):** Trong hồi quy tuyến tính một biến, w là hệ số nhân với đầu vào x trong dự đoán ŷ=wx+b. Nó là hệ số của biến giải thích trong mô hình tuyến tính.

**Vai trò trong ML:** w quyết định độ dốc: thay đổi x một đơn vị làm dự đoán thay đổi w đơn vị. Khi huấn luyện, w được ước lượng từ dữ liệu.

**Ví dụ:** Nếu w=2 và b=1, x=3 thì ŷ=7.

**Code minh họa:**

```python
w, x, b = 2.0, 3.0, 1.0
y_hat = w * x + b
print(y_hat)  # 7.0
```

**Nguồn:** [Wikipedia: Linear regression](https://en.wikipedia.org/wiki/Linear_regression)

### Bias (độ lệch b) có vai trò gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Tham số mô hình

**Định nghĩa (Wikipedia):** Trong hồi quy tuyến tính một biến, b là hệ số chặn (intercept) trong ŷ=wx+b: giá trị dự đoán khi x=0.

**Vai trò trong ML:** Bias cho phép đường dự đoán dịch lên hoặc xuống thay vì bị buộc đi qua gốc tọa độ. Trong mô hình nhiều biến, nó vẫn là hạng cộng thêm độc lập với các đầu vào.

**Ví dụ:** Với w=2, b=1, khi x=0 thì ŷ=1.

**Code minh họa:**

```python
w, b, x = 2.0, 1.0, 0.0
y_hat = w * x + b
print(y_hat)  # 1.0
```

**Nguồn:** [Wikipedia: Linear regression](https://en.wikipedia.org/wiki/Linear_regression)

### Learning rate (lr) là gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Tối ưu hóa

**Định nghĩa (Wikipedia):** Learning rate là tham số điều chỉnh thuật toán tối ưu hóa, quyết định độ dài bước ở mỗi lần cập nhật khi tiến về cực tiểu của loss.

**Vai trò trong ML:** Gradient chọn hướng cập nhật; learning rate chọn độ dài bước. Giá trị quá lớn có thể vượt qua cực tiểu, còn giá trị quá nhỏ làm hội tụ chậm.

**Ví dụ:** Nếu gradient là 2 và lr=0.1, tham số giảm 0.2 theo quy tắc cập nhật.

**Code minh họa:**

```python
theta, grad, lr = 3.0, 2.0, 0.1
theta -= lr * grad
print(theta)  # 2.8
```

**Nguồn:** [Wikipedia: Learning rate](https://en.wikipedia.org/wiki/Learning_rate)

### Epoch trong huấn luyện ML là gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Huấn luyện

**Định nghĩa (Wikipedia):** Trong ngữ cảnh stochastic gradient, Wikipedia gọi nhiều lượt đi qua dữ liệu là các pass/cycle/epoch. Ở cách huấn luyện theo batch phổ biến, một epoch thường là một lượt xử lý toàn bộ tập huấn luyện.

**Vai trò trong ML:** Epoch đếm số lượt dữ liệu được dùng trong huấn luyện; nó không tự nói mô hình đã học tốt hay loss đã hội tụ.

**Ví dụ:** Hai epoch nghĩa là vòng lặp huấn luyện đi qua dữ liệu hai lượt.

**Code minh họa:**

```python
data = [(1, 2), (2, 4)]
for epoch in range(2):
    for x, y in data:
        pass  # xử lý một mẫu
```

**Nguồn:** [Wikipedia: Online machine learning](https://en.wikipedia.org/wiki/Online_machine_learning); [Wikipedia: Learning curve (machine learning)](https://en.wikipedia.org/wiki/Learning_curve_(machine_learning))

### Backpropagation (lan truyền ngược) là gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Lan truyền ngược

**Định nghĩa (Wikipedia):** Backpropagation là thuật toán tính gradient của hàm mất mát theo các tham số trong mạng neural bằng cách truyền đạo hàm ngược qua các phép tính và áp dụng chain rule.

**Vai trò trong ML:** Backpropagation tính gradient; nó không đồng nghĩa với việc cập nhật trọng số. Optimizer như gradient descent dùng gradient đó để cập nhật tham số.

**Ví dụ:** Nếu y=a·x và loss=(y−t)² thì chain rule ghép đạo hàm loss theo y với đạo hàm y theo a để tìm d(loss)/da.

**Code minh họa:**

```python
a, x, target = 2.0, 3.0, 5.0
y = a * x
dloss_dy = 2 * (y - target)
dy_da = x
dloss_da = dloss_dy * dy_da
```

**Nguồn:** [Wikipedia: Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)

### Chain rule (quy tắc dây chuyền) dùng để làm gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Đạo hàm và giải tích

**Định nghĩa (Wikipedia):** Quy tắc dây chuyền tính đạo hàm của hàm hợp. Nếu z phụ thuộc y và y phụ thuộc x, đạo hàm của z theo x bằng đạo hàm z theo y nhân đạo hàm y theo x.

**Vai trò trong ML:** Backpropagation dùng quy tắc này lặp qua từng phép tính để chuyển gradient từ đầu ra về các đầu vào và tham số.

**Ví dụ:** Nếu y=x² và z=3y, tại x=2: dz/dx=(dz/dy)(dy/dx)=3×4=12.

**Code minh họa:**

```python
x = 2.0
dy_dx = 2 * x
dz_dy = 3.0
dz_dx = dz_dy * dy_dx
print(dz_dx)  # 12.0
```

**Nguồn:** [Wikipedia: Chain rule](https://en.wikipedia.org/wiki/Chain_rule)

### Autograd (đạo hàm tự động) là gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Đạo hàm tự động

**Định nghĩa (Wikipedia):** Automatic differentiation là nhóm kỹ thuật tính đạo hàm của hàm được biểu diễn bằng chương trình bằng cách áp dụng chain rule qua các phép toán cơ bản, với độ chính xác số học của máy.

**Vai trò trong ML:** Autograd ghi nhận hoặc xử lý phép tính để tạo gradient; khác với sai phân hữu hạn, vốn ước lượng đạo hàm bằng cách lấy hiệu các giá trị gần nhau.

**Ví dụ:** Tensor có requires_grad=True sẽ giữ quan hệ phép tính; gọi backward() tính gradient.

**Code minh họa:**

```python
import torch
x = torch.tensor(3.0, requires_grad=True)
y = x * x
y.backward()
print(x.grad.item())  # 6.0
```

**Nguồn:** [Wikipedia: Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)

### Vector là gì trong hình học Euclid?

**Nhóm:** Thuật ngữ · **Chủ đề:** Đại số tuyến tính

**Định nghĩa (Wikipedia):** Vector Euclid là đại lượng có độ lớn và hướng; thường được hình dung như một mũi tên biểu diễn độ dời. Trong không gian vector trừu tượng, khái niệm vector được tổng quát hóa thành phần tử của không gian đó.

**Vai trò trong ML:** Trong các ví dụ ML, vector thường chứa các giá trị đặc trưng của một mẫu, đầu ra của một lớp hoặc các thành phần gradient.

**Ví dụ:** Vector [3,4] biểu diễn hướng và có độ dài 5.

**Code minh họa:**

```python
v = [3.0, 4.0]
length = (sum(x*x for x in v)) ** 0.5
print(length)  # 5.0
```

**Nguồn:** [Wikipedia: Euclidean vector](https://en.wikipedia.org/wiki/Euclidean_vector)

### Ma trận là gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Đại số tuyến tính

**Định nghĩa (Wikipedia):** Ma trận là một mảng chữ nhật các phần tử được sắp theo hàng và cột. Một ma trận có m hàng và n cột biểu diễn ánh xạ giữa các vector có số chiều tương ứng.

**Vai trò trong ML:** Trong ML, ma trận thường chứa trọng số của lớp tuyến tính; phép nhân ma trận thực hiện phép biến đổi lên vector đầu vào.

**Ví dụ:** Ma trận 2×2 có hai hàng và hai cột.

**Code minh họa:**

```python
A = [[1, 2], [3, 4]]
rows, cols = len(A), len(A[0])
print(rows, cols)  # 2 2
```

**Nguồn:** [Wikipedia: Matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics)); [Wikipedia: Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication)

### Dot product (tích vô hướng) là gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Đại số tuyến tính

**Định nghĩa (Wikipedia):** Với hai vector cùng số chiều, tích vô hướng là tổng các tích của những thành phần nằm cùng vị trí; kết quả là một scalar.

**Vai trò trong ML:** Dùng để tính tổng có trọng số, độ tương đồng hướng và phép chiếu vector.

**Ví dụ:** [1,2]·[3,4]=1×3+2×4=11.

**Code minh họa:**

```python
a, b = [1, 2], [3, 4]
dot = sum(x*y for x, y in zip(a, b))
print(dot)  # 11
```

**Nguồn:** [Wikipedia: Dot product](https://en.wikipedia.org/wiki/Dot_product)

### Norm / magnitude (chuẩn Euclid) là gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Đại số tuyến tính

**Định nghĩa (Wikipedia):** Chuẩn Euclid của vector là căn bậc hai của tích vô hướng vector với chính nó; về hình học, đó là độ dài Euclid của vector.

**Vai trò trong ML:** Dùng để đo độ lớn vector và chuẩn hóa vector. Cần nói rõ chuẩn nào khi làm việc với các chuẩn khác nhau.

**Ví dụ:** ||[3,4]||₂=√(3²+4²)=5.

**Code minh họa:**

```python
v = [3, 4]
norm = sum(x*x for x in v) ** 0.5
print(norm)  # 5.0
```

**Nguồn:** [Wikipedia: Euclidean norm](https://en.wikipedia.org/wiki/Euclidean_norm)

### Cosine similarity đo điều gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Độ tương đồng vector

**Định nghĩa (Wikipedia):** Cosine similarity là cosine của góc giữa hai vector khác vector 0, bằng tích vô hướng chia cho tích độ dài hai vector.

**Vai trò trong ML:** Nó đo độ tương đồng về hướng; chuẩn hóa theo độ dài làm cho vector cùng hướng có điểm cao dù độ lớn khác nhau.

**Ví dụ:** [1,0] và [5,0] cùng hướng nên cosine similarity bằng 1.

**Code minh họa:**

```python
def dot(a, b):
    return sum(x*y for x, y in zip(a, b))
def norm(a):
    return sum(x*x for x in a) ** 0.5
a, b = [1, 0], [5, 0]
score = dot(a, b) / (norm(a) * norm(b))
print(score)  # 1.0
```

**Nguồn:** [Wikipedia: Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

### Hessian là gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Đạo hàm và giải tích

**Định nghĩa (Wikipedia):** Hessian của hàm vô hướng nhiều biến là ma trận vuông gồm các đạo hàm riêng bậc hai. Khi các đạo hàm hỗn hợp liên tục, Hessian đối xứng.

**Vai trò trong ML:** Hessian mô tả độ cong cục bộ: các phần tử đường chéo là độ cong theo từng biến, còn phần tử ngoài đường chéo thể hiện đạo hàm hỗn hợp.

**Ví dụ:** Với f(x,y)=x²+y², Hessian là [[2,0],[0,2]].

**Code minh họa:**

```python
H = [[2, 0],
     [0, 2]]  # Hessian của x**2 + y**2
```

**Nguồn:** [Wikipedia: Hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix)

### ReLU là gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Mạng neural

**Định nghĩa (Wikipedia):** Rectified linear unit là hàm kích hoạt f(x)=max(0,x): giữ nguyên đầu vào dương và cho 0 với đầu vào không dương.

**Vai trò trong ML:** Trong mạng neural, ReLU tạo phép biến đổi phi tuyến; đạo hàm bằng 1 khi đầu vào dương và bằng 0 khi đầu vào âm (tại 0 thường chọn một quy ước).

**Ví dụ:** ReLU(-2)=0, ReLU(3)=3.

**Code minh họa:**

```python
def relu(x):
    return max(0.0, x)
print(relu(-2), relu(3))  # 0.0 3
```

**Nguồn:** [Wikipedia: Rectified linear unit](https://en.wikipedia.org/wiki/Rectified_linear_unit)

### Trị riêng và vector riêng liên hệ thế nào?

**Nhóm:** Thuật ngữ · **Chủ đề:** Trị riêng và vector riêng

**Định nghĩa (Wikipedia):** Với ma trận A, vector riêng v khác 0 thỏa Av=λv. Số λ là trị riêng tương ứng: phép biến đổi chỉ co giãn vector đó hoặc đảo chiều nó, không đổi phương.

**Vai trò trong ML:** Các trị riêng và vector riêng mô tả những hướng đặc biệt của phép biến đổi tuyến tính.

**Ví dụ:** Với A=[[2,0],[0,3]], vector [1,0] là vector riêng có trị riêng 2.

**Code minh họa:**

```python
A = [[2, 0], [0, 3]]
v = [1, 0]
Av = [sum(r[i]*v[i] for i in range(2)) for r in A]
print(Av)  # [2, 0]
```

**Nguồn:** [Wikipedia: Eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)

### Định thức bằng 0 cho biết điều gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Định thức và ma trận nghịch đảo

**Định nghĩa (Wikipedia):** Định thức là scalar gắn với ma trận vuông. Với ma trận biểu diễn biến đổi tuyến tính thực, trị tuyệt đối của định thức là hệ số thay đổi thể tích; dấu cho biết định hướng có bị đảo hay không.

**Vai trò trong ML:** Định thức bằng 0 nghĩa là biến đổi làm sụp chiều, ma trận suy biến và không có nghịch đảo.

**Ví dụ:** Hai hàng [1,2] và [2,4] phụ thuộc tuyến tính, nên ma trận có định thức 0.

**Code minh họa:**

```python
a, b, c, d = 1, 2, 2, 4
det = a*d - b*c
print(det)  # 0
```

**Nguồn:** [Wikipedia: Determinant](https://en.wikipedia.org/wiki/Determinant)

### Độc lập tuyến tính nghĩa là gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Đại số tuyến tính

**Định nghĩa (Wikipedia):** Một tập vector độc lập tuyến tính nếu tổ hợp tuyến tính bằng vector 0 chỉ có nghiệm mà mọi hệ số đều bằng 0. Tương đương, không vector nào là tổ hợp tuyến tính của các vector còn lại.

**Vai trò trong ML:** Độc lập tuyến tính cho biết tập vector chứa những hướng riêng biệt, không dư thừa tuyến tính.

**Ví dụ:** [1,0] và [0,1] độc lập; [2,0] là bội của [1,0], nên cặp [1,0],[2,0] phụ thuộc.

**Code minh họa:**

```python
a, b = [1, 0], [0, 1]
det = a[0]*b[1] - a[1]*b[0]
independent = det != 0
print(independent)  # True
```

**Nguồn:** [Wikipedia: Linear independence](https://en.wikipedia.org/wiki/Linear_independence)

### Gram–Schmidt làm gì?

**Nhóm:** Thuật ngữ · **Chủ đề:** Đại số tuyến tính

**Định nghĩa (Wikipedia):** Gram–Schmidt là quy trình biến một tập vector độc lập trong không gian tích vô hướng thành tập trực chuẩn sinh cùng không gian con.

**Vai trò trong ML:** Mỗi bước loại khỏi vector mới các hình chiếu lên những hướng đã xử lý; sau đó chuẩn hóa vector còn lại.

**Ví dụ:** Từ [1,1] sau khi loại thành phần chiếu lên [1,0], còn hướng [0,1].

**Code minh họa:**

```python
u = [1.0, 0.0]
v = [1.0, 1.0]
dot = sum(a*b for a, b in zip(v, u))
proj = [dot*x for x in u]
w = [v[i]-proj[i] for i in range(2)]
print(w)  # [0.0, 1.0]
```

**Nguồn:** [Wikipedia: Gram–Schmidt process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)

## Công thức

### Công thức tích vô hướng là gì? Giải thích các thành phần.

**Nhóm:** Công thức · **Chủ đề:** Vector và hình học

**Công thức:** `a·b = Σᵢ aᵢbᵢ`

**Giải thích thành phần / vai trò:** a và b là hai vector cùng số chiều; aᵢ và bᵢ là thành phần thứ i. Σᵢ yêu cầu cộng các tích qua mọi vị trí. Tích vô hướng trả về một scalar.

**Ví dụ:** Với a=[1,2], b=[3,4]: 1×3+2×4=11.

**Code minh họa:**

```python
a, b = [1, 2], [3, 4]
result = sum(x*y for x, y in zip(a, b))
```

**Nguồn:** [Wikipedia: Dot product](https://en.wikipedia.org/wiki/Dot_product)

### Công thức chuẩn L2 của vector là gì? Giải thích các thành phần.

**Nhóm:** Công thức · **Chủ đề:** Vector và hình học

**Công thức:** `||v||₂ = √(Σᵢ vᵢ²)`

**Giải thích thành phần / vai trò:** vᵢ là thành phần thứ i; bình phương từng thành phần rồi cộng; cuối cùng lấy căn bậc hai. Kết quả là độ dài Euclid của v.

**Ví dụ:** ||[3,4]||₂=√(9+16)=5.

**Code minh họa:**

```python
v = [3, 4]
norm = sum(x*x for x in v) ** 0.5
```

**Nguồn:** [Wikipedia: Euclidean norm](https://en.wikipedia.org/wiki/Euclidean_norm)

### Công thức chuẩn hóa vector là gì? Điều kiện nào cần có?

**Nhóm:** Công thức · **Chủ đề:** Vector và hình học

**Công thức:** `v̂ = v / ||v||₂, với v ≠ 0.`

**Giải thích thành phần / vai trò:** v là vector ban đầu; ||v||₂ là độ dài; v̂ là vector đơn vị thu được. Chia cho độ dài giữ hướng nhưng đưa độ dài về 1. Vector 0 không chuẩn hóa được.

**Ví dụ:** [3,4]/5=[0.6,0.8], có độ dài 1.

**Code minh họa:**

```python
v = [3.0, 4.0]
norm = sum(x*x for x in v) ** 0.5
v_hat = [x/norm for x in v]
```

**Nguồn:** [Wikipedia: Unit vector](https://en.wikipedia.org/wiki/Unit_vector); [Wikipedia: Euclidean norm](https://en.wikipedia.org/wiki/Euclidean_norm)

### Công thức cosine similarity là gì? Mẫu số có vai trò gì?

**Nhóm:** Công thức · **Chủ đề:** Độ tương đồng vector

**Công thức:** `cos(a,b) = (a·b)/(||a||₂ ||b||₂), với a,b khác vector 0.`

**Giải thích thành phần / vai trò:** Tử số là tích vô hướng; mẫu số là tích độ dài. Chia cho độ dài loại ảnh hưởng độ lớn và cho cosine góc giữa hai vector.

**Ví dụ:** Hai vector cùng hướng như [1,0] và [5,0] có giá trị 1.

**Code minh họa:**

```python
def dot(a, b):
    return sum(x*y for x, y in zip(a, b))
def norm(a):
    return sum(x*x for x in a) ** 0.5
a, b = [1, 0], [5, 0]
score = dot(a, b)/(norm(a)*norm(b))
```

**Nguồn:** [Wikipedia: Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

### Công thức chiếu vector a lên b là gì? Giải thích các thành phần.

**Nhóm:** Công thức · **Chủ đề:** Vector và phép chiếu

**Công thức:** `proj_b(a) = ((a·b)/(b·b))b, với b ≠ 0.`

**Giải thích thành phần / vai trò:** Hệ số (a·b)/(b·b) là độ dài có dấu của thành phần theo hướng b; nhân hệ số đó với b cho vector hình chiếu vuông góc.

**Ví dụ:** Chiếu [3,2] lên [1,0] được [3,0].

**Code minh họa:**

```python
a, b = [3.0, 2.0], [1.0, 0.0]
scale = sum(x*y for x, y in zip(a,b)) / sum(y*y for y in b)
projection = [scale*y for y in b]
```

**Nguồn:** [Wikipedia: Vector projection](https://en.wikipedia.org/wiki/Vector_projection)

### Công thức sai phân trung tâm để ước lượng đạo hàm là gì?

**Nhóm:** Công thức · **Chủ đề:** Đạo hàm số

**Công thức:** `f′(x) ≈ [f(x+h)−f(x−h)]/(2h), với h nhỏ khác 0.`

**Giải thích thành phần / vai trò:** x là điểm cần ước lượng; h là bước dịch. Hai giá trị đối xứng quanh x giúp ước lượng độ dốc tại x. Đây là phép xấp xỉ số, không phải định nghĩa chính xác của đạo hàm.

**Ví dụ:** Với f(x)=x² và x=3, kết quả tiến gần 6 khi h nhỏ.

**Code minh họa:**

```python
def derivative(f, x, h=1e-5):
    return (f(x+h)-f(x-h))/(2*h)
print(derivative(lambda x: x*x, 3.0))
```

**Nguồn:** [Wikipedia: Numerical differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation); [Wikipedia: Derivative](https://en.wikipedia.org/wiki/Derivative)

### Với f(x,y)=x²+3xy+y², gradient là gì? Tính tại (1,2).

**Nhóm:** Công thức · **Chủ đề:** Gradient và tối ưu hóa

**Công thức:** `∇f(x,y) = [∂f/∂x, ∂f/∂y] = [2x+3y, 3x+2y].`

**Giải thích thành phần / vai trò:** x,y là tọa độ; mỗi thành phần là đạo hàm riêng theo tọa độ đó. Gradient gom tốc độ thay đổi theo mọi trục.

**Ví dụ:** Tại (1,2): [2+6, 3+4]=[8,7].

**Code minh họa:**

```python
x, y = 1.0, 2.0
grad = [2*x + 3*y, 3*x + 2*y]
print(grad)  # [8.0, 7.0]
```

**Nguồn:** [Wikipedia: Gradient](https://en.wikipedia.org/wiki/Gradient); [Wikipedia: Partial derivative](https://en.wikipedia.org/wiki/Partial_derivative)

### Công thức gradient descent là gì? Giải thích θ và η.

**Nhóm:** Công thức · **Chủ đề:** Tối ưu hóa

**Công thức:** `θₜ₊₁ = θₜ − η∇L(θₜ).`

**Giải thích thành phần / vai trò:** θ là vector tham số; L là loss; ∇L là gradient loss theo θ; η là learning rate. Trừ gradient để bước theo hướng giảm nhanh nhất cục bộ với bước đủ nhỏ.

**Ví dụ:** Nếu θ=3, gradient=2, η=0.1 thì θ mới là 2.8.

**Code minh họa:**

```python
theta, grad, eta = 3.0, 2.0, 0.1
theta = theta - eta*grad
```

**Nguồn:** [Wikipedia: Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent); [Wikipedia: Learning rate](https://en.wikipedia.org/wiki/Learning_rate)

### Trong hồi quy tuyến tính, công thức dự đoán, sai số và MSE là gì?

**Nhóm:** Công thức · **Chủ đề:** Hồi quy tuyến tính

**Công thức:** `ŷᵢ = wxᵢ+b; eᵢ=ŷᵢ−yᵢ; MSE = (1/N)Σᵢ(ŷᵢ−yᵢ)².`

**Giải thích thành phần / vai trò:** w là hệ số góc; b là hệ số chặn; xᵢ là đầu vào; yᵢ là đáp án; ŷᵢ là dự đoán; N là số mẫu. MSE bình phương sai số để dấu âm/dương không triệt tiêu rồi lấy trung bình.

**Ví dụ:** Hai sai số 1 và -3 cho MSE=(1²+(-3)²)/2=5.

**Code minh họa:**

```python
predictions, targets = [2.0, 4.0], [1.0, 7.0]
mse = sum((p-y)**2 for p, y in zip(predictions, targets))/len(targets)
```

**Nguồn:** [Wikipedia: Linear regression](https://en.wikipedia.org/wiki/Linear_regression); [Wikipedia: Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)

### Quy tắc dây chuyền có công thức nào?

**Nhóm:** Công thức · **Chủ đề:** Đạo hàm và lan truyền ngược

**Công thức:** `Nếu z=f(y) và y=g(x), thì dz/dx=(dz/dy)(dy/dx).`

**Giải thích thành phần / vai trò:** dz/dy và dy/dx là đạo hàm cục bộ ở hai mắt xích. Nhân chúng để tìm ảnh hưởng tổng thể từ x tới z; backpropagation lặp phép này dọc đồ thị.

**Ví dụ:** y=x², z=3y, tại x=2: dz/dx=3×4=12.

**Code minh họa:**

```python
x = 2.0
dy_dx = 2*x
dz_dy = 3.0
dz_dx = dz_dy*dy_dx
```

**Nguồn:** [Wikipedia: Chain rule](https://en.wikipedia.org/wiki/Chain_rule)

### Nếu z=a+b, gradient cục bộ theo a và b là gì?

**Nhóm:** Công thức · **Chủ đề:** Đạo hàm tự động

**Công thức:** `∂z/∂a=1 và ∂z/∂b=1, khi z=a+b.`

**Giải thích thành phần / vai trò:** Mỗi đầu vào đóng góp trực tiếp một đơn vị vào tổng. Khi lan truyền gradient ngược, gradient phía sau được chuyển nguyên vẹn tới cả a và b.

**Ví dụ:** Nếu a=2,b=3 thì z=5; tăng a một đơn vị làm z tăng một đơn vị.

**Code minh họa:**

```python
a, b = 2.0, 3.0
z = a + b
dz_da = dz_db = 1.0
```

**Nguồn:** [Wikipedia: Differentiation rules](https://en.wikipedia.org/wiki/Differentiation_rules#Sum_rule); [Wikipedia: Chain rule](https://en.wikipedia.org/wiki/Chain_rule)

### Nếu z=a×b, gradient cục bộ theo a và b là gì?

**Nhóm:** Công thức · **Chủ đề:** Đạo hàm tự động

**Công thức:** `∂z/∂a=b và ∂z/∂b=a, khi z=ab.`

**Giải thích thành phần / vai trò:** Đạo hàm theo một thừa số bằng thừa số còn lại. Trong backpropagation, nhân đạo hàm cục bộ này với gradient đi từ phía sau.

**Ví dụ:** Với a=2,b=3: ∂z/∂a=3 và ∂z/∂b=2.

**Code minh họa:**

```python
a, b = 2.0, 3.0
z = a*b
dz_da, dz_db = b, a
```

**Nguồn:** [Wikipedia: Product rule](https://en.wikipedia.org/wiki/Product_rule); [Wikipedia: Chain rule](https://en.wikipedia.org/wiki/Chain_rule)

### Khai triển Taylor bậc hai quanh x₀ có dạng nào?

**Nhóm:** Công thức · **Chủ đề:** Xấp xỉ Taylor

**Công thức:** `f(x₀+h) ≈ f(x₀)+f′(x₀)h+½f″(x₀)h².`

**Giải thích thành phần / vai trò:** x₀ là điểm khai triển; h là độ dịch; f′ là độ dốc; f″ là độ cong. Đây là xấp xỉ cục bộ và thường kém chính xác khi đi xa x₀.

**Ví dụ:** Với sin tại x₀=0, bậc hai cho sin(h)≈h vì sin(0)=0, cos(0)=1, sin″(0)=0.

**Code minh họa:**

```python
import math
x0, h = 0.0, 0.2
approx = math.sin(x0) + math.cos(x0)*h - 0.5*math.sin(x0)*h*h
```

**Nguồn:** [Wikipedia: Taylor theorem](https://en.wikipedia.org/wiki/Taylor_theorem)

### Ma trận Hessian 2 biến có cấu trúc nào và các phần tử nói gì?

**Nhóm:** Công thức · **Chủ đề:** Hessian và độ cong

**Công thức:** `H = [[fₓₓ, fₓᵧ], [fᵧₓ, fᵧᵧ]]. Nếu đạo hàm hỗn hợp liên tục thì fₓᵧ=fᵧₓ.`

**Giải thích thành phần / vai trò:** fₓₓ và fᵧᵧ là độ cong theo từng tọa độ; hai phần tử ngoài đường chéo là đạo hàm hỗn hợp, mô tả tương tác giữa hai hướng.

**Ví dụ:** Với f=x²+y², H=[[2,0],[0,2]].

**Code minh họa:**

```python
H = [[2.0, 0.0],
     [0.0, 2.0]]  # Hessian của x**2 + y**2
```

**Nguồn:** [Wikipedia: Hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix)

### Công thức trị riêng và vector riêng là gì?

**Nhóm:** Công thức · **Chủ đề:** Trị riêng và vector riêng

**Công thức:** `Av=λv, với v≠0.`

**Giải thích thành phần / vai trò:** A là ma trận; v là vector riêng; λ là trị riêng. Sau biến đổi, v chỉ co giãn theo λ hoặc đảo chiều nếu λ âm.

**Ví dụ:** Với A=diag(2,3), v=[1,0] thỏa Av=2v.

**Code minh họa:**

```python
A, v = [[2, 0], [0, 3]], [1, 0]
Av = [sum(row[i]*v[i] for i in range(2)) for row in A]
print(Av)  # [2, 0]
```

**Nguồn:** [Wikipedia: Eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)

### Công thức định thức ma trận 2×2 là gì? Khi nào ma trận suy biến?

**Nhóm:** Công thức · **Chủ đề:** Định thức

**Công thức:** `det([[a,b],[c,d]])=ad−bc.`

**Giải thích thành phần / vai trò:** a,b,c,d là các phần tử của ma trận. Nếu ad−bc=0 thì ma trận suy biến: không khả nghịch và phép biến đổi làm sụp một chiều.

**Ví dụ:** det([[1,2],[2,4]])=1×4−2×2=0.

**Code minh họa:**

```python
A = [[1, 2], [2, 4]]
det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
```

**Nguồn:** [Wikipedia: Determinant](https://en.wikipedia.org/wiki/Determinant)

### Công thức nghịch đảo ma trận 2×2 là gì?

**Nhóm:** Công thức · **Chủ đề:** Ma trận nghịch đảo

**Công thức:** `Nếu A=[[a,b],[c,d]] và ad−bc≠0, thì A⁻¹=(1/(ad−bc))[[d,−b],[−c,a]].`

**Giải thích thành phần / vai trò:** ad−bc là định thức; điều kiện khác 0 bảo đảm nghịch đảo tồn tại. A⁻¹ hoàn tác phép biến đổi của A.

**Ví dụ:** Với A=[[1,2],[3,4]], det=-2 và A⁻¹=[[-2,1],[1.5,-0.5]].

**Code minh họa:**

```python
A = [[1.0, 2.0], [3.0, 4.0]]
a,b = A[0]
c,d = A[1]
det = a*d-b*c
A_inv = [[d/det, -b/det], [-c/det, a/det]]
```

**Nguồn:** [Wikipedia: Invertible matrix](https://en.wikipedia.org/wiki/Invertible_matrix); [Wikipedia: Determinant](https://en.wikipedia.org/wiki/Determinant)

### Ma trận quay 2D một góc θ là gì? θ được đo bằng đơn vị nào?

**Nhóm:** Công thức · **Chủ đề:** Biến đổi tuyến tính

**Công thức:** `R(θ)=[[cosθ,−sinθ],[sinθ,cosθ]], với θ tính bằng radian theo quy ước quay dương ngược chiều kim đồng hồ.`

**Giải thích thành phần / vai trò:** Nhân R(θ) với vector cột quay vector quanh gốc tọa độ mà không đổi độ dài.

**Ví dụ:** θ=π/2 quay vector [1,0] thành [0,1].

**Code minh họa:**

```python
import math
theta = math.pi/2
c, s = math.cos(theta), math.sin(theta)
R = [[c, -s], [s, c]]
```

**Nguồn:** [Wikipedia: Rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix)

## Ghi chú về thuật ngữ epoch

Wikipedia không có trang riêng phù hợp cho `epoch` theo nghĩa lượt huấn luyện ML. Thẻ dùng nội dung liên quan trong bài [Online machine learning](https://en.wikipedia.org/wiki/Online_machine_learning), nơi “passes/cycles/epochs” được dùng cho các lượt xử lý dữ liệu; định nghĩa trong deck được ghi rõ là cách dùng trong ngữ cảnh huấn luyện, không gán với trang về epoch lịch sử.

## Lưu ý về code

Code minh họa được viết để làm rõ từng khái niệm; các đoạn PyTorch cần cài PyTorch. Những snippet còn lại chỉ dùng Python chuẩn. Phần định nghĩa được Việt hóa từ nguồn liên kết; ví dụ và vai trò học máy là phần giải thích biên soạn riêng.

## Bổ sung: Phép nhân ma trận (matmul)

**Định nghĩa (Wikipedia):** Phép nhân ma trận nhận hai ma trận phù hợp kích thước và tạo một ma trận mới. Mỗi ô kết quả là tích vô hướng giữa hàng tương ứng của ma trận thứ nhất và cột tương ứng của ma trận thứ hai.

**Vai trò trong ML:** Một lớp tuyến tính thường nhân ma trận trọng số với vector đầu vào, sau đó cộng bias.

**Ví dụ:** Với A=`[[1,2],[3,4]]` và B=`[[5,6],[7,8]]`, ô C[0][0] là `1×5+2×7=19`.

**Code minh họa:**

```python
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = [[sum(A[i][k]*B[k][j] for k in range(2))
      for j in range(2)] for i in range(2)]
print(C)  # [[19, 22], [43, 50]]
```

**Nguồn định nghĩa:** [Wikipedia: Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication).

Thẻ MatMul trong file `.apkg` có thêm minh họa tương tác được nhúng từ `visual/matmul-visualization.html`; file HTML riêng không phải định dạng deck để import trực tiếp.
