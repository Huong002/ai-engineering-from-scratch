"""

"""
class Value:
    
    def __init__(self , data, children=(),  op=''):
        self.data = data
        self.grad = 0.0 # gradient: theo doii gia  tri  anh  huong den ket qua cuoi  cung  
        self._backward = lambda: None
        self.prev = set(children)
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
        other =  other if isinstance(other, Value) else Value(other) # isinstane kime tra xem  other co phai doi  tuong cua VAlue? 
        out =  Value(self.data + other.data, (self, other), '+' )
        def _backward(): 
            self.grad += out.grad
            other.grad += out.graond
        out._backward = _backward
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
