"""

"""
class Value:
    
    def __init__(self, data, children=(), op=''):
        # data là giá trị số mà đối tượng Value đang đại diện.
        self.data = data
        # grad lưu gradient: mức độ data ảnh hưởng đến kết quả cuối cùng; ban đầu chưa có nên bằng 0.
        self.grad = 0.0
        # Hàm lan truyền gradient mặc định chưa làm gì; mỗi phép toán sẽ gán quy tắc riêng.
        self._backward = lambda: None
        # Lưu các Value đứng trước trong đồ thị tính toán; set giúp mỗi nút cha chỉ xuất hiện một lần.
        self.prev = set(children)
        # Lưu phép toán đã tạo ra giá trị này (ví dụ: '+', '*') để theo dõi đồ thị tính toán.
        self._op = op
        
    def __repr__(self):
        return f"Value(data={self.data:.4f} , grad={self.grad:.4f})"

    def __add__(self, other):
        """Cộng giá trị này với một Value khác hoặc một số.

        Args:
            other: Giá trị hoặc số cần cộng.

        Returns:
            Value: Kết quả phép cộng, kèm quy tắc lan truyền gradient.
        """
        # Nếu other đã là Value thì dùng luôn; nếu là số thường thì bọc thành Value.
        other = other if isinstance(other, Value) else Value(other)
        # Tính tổng hai giá trị, đồng thời ghi nhớ hai nút cha và phép toán '+' trong đồ thị.
        out = Value(self.data + other.data, (self, other), '+')

        # Khi lan truyền ngược, gradient của tổng được cộng vào gradient của cả hai số hạng.
        # Vì d(a + b)/da = 1 và d(a + b)/db = 1 nên mỗi bên nhận nguyên out.grad.
        def _backward():
            # += để cộng dồn nếu self còn tham gia vào những phép tính khác.
            self.grad += out.grad
            # Tương tự, other cũng nhận nguyên gradient của kết quả phép cộng.
            other.grad += out.grad

        # Gắn quy tắc lan truyền ngược vừa định nghĩa vào Value kết quả.
        out._backward = _backward
        # Trả Value mới để có thể tiếp tục kết hợp với các phép toán khác.
        return out

        
    def __mul__(self, other):
        """Nhân giá trị này với một Value khác hoặc một số.

        Args:
            other: Giá trị hoặc số cần nhân.

        Returns:
            Value: Kết quả phép nhân, kèm quy tắc lan truyền gradient.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other), '*')
        def _backward():
            self.grad +=  other.data  * out.grad
            other.grad +=  self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """Áp dụng hàm ReLU lên giá trị này.

        Returns:
            Value: Giá trị đầu vào nếu dương, ngược lại là 0; kèm quy tắc lan truyền gradient của ReLU.
        """
        out = Value(max(0, self.data), (self,),'relu')
        def _backward():
            self.grad  += (1.0 if out.data >0 else 0.0 ) * out.grad
        out._backward = _backward
        return out
   
   
    def backward(self):
        topo = []
        vistied = set()
        def build_topo(v):
            if v not in vistied:
                vistied.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
    