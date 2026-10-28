class Vector:
    
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

        
    def __repr__(self):
        return f"Vector({self.data})"

        
    def __add__(self, other):
        if self.size != other.size:
            raise ValueError("Vectors must have the same size")
        return Vector([a+b for a, b in zip(self.data, other.data)])

        
    def __sub__(self, other):
        if self.size != other.size:
            raise ValueError("Vectors must have the same size")
        return Vector([a-b for a , b in zip(self.data, other.data)])

        
    def __mul__(self, scalar):
        return Vector([x * scalar for x in  self.data])  

        
        
    def dot(self, other):
        if self.size != other.size:
            raise ValueError("Vectors must have the same size")
        return sum(a * b for a, b in zip(self.data, other.data))

        
    def magnitude(self):
        return sum(x ** 2  for x in self.data) **  0.5 
    
    
    
class Matrix:

    """
    for i in  range(self.rows): lan luot  cchon tung hang  ccuua ma traan
    for j in  range(other.cols):  voi  moi hang do , tinh tung cot  cua tran other
    for k in range(self.cols): nhacn cac phhan tu  tuong uung trong hang va cot, roi cong lai
    """
    def __init__(self, data):
        self.data = [list(row) for  row in data]
        if not self.data or not self.data[0]:
            raise ValueError("Matrix must have at least one row and one column")
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        if any(len(row) != self.cols for row in self.data):
            raise ValueError("All matrix rows must have the same number of columns")
        self.shape = (self.rows, self.cols)

        
    def __repr__(self):
        """quyet dinh cach doi tuong duo bieu dien khi hien thi trong python"""
        rows_str = "\n ".join(str(row)  for row in self.data)
        return f"Matrix({self.shape}): \n {rows_str}"
    
    
    
    def __add__(self, other):

        """phep cong  2 ma tran cung vi tri d"""
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same shape")
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])


        
    def __sub__(self, other):

        """phep tru  2 ma tran cung vi tri d"""
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same shape")
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar ):
        """Nhan ma tran cho  mot sso"""
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def element_wise_multiply(self,  other):
        if self.shape != other.shape:
            raise ValueError("Matrices must have the same shape")
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])
        
    def matmul(self, other):
        if self.cols != other.rows:
            raise ValueError("Left matrix columns must match right matrix rows")
        return Matrix([
            [
            sum(self.data[i][k] * other.data[k][j]  for k in range(self.cols))
            for j  in  range(other.cols)
            ]
            for i in range(self.rows)
        ])

        
    def transpose(self):
        return Matrix([
            [self.data[j][i] for  j in  range(self.rows)]
            for i in range(self.cols)
        ])

    def  determinanr(self):
        if self.rows != self.cols:
            raise ValueError("Determinant is defined only for square matrices")
        if self.shape == (1,1):
            return self.data[0][0]
        if self.shape == (2,2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0 
        for j  in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for  k in  range(self.cols) if  k != j]
                for i in  range(1, self.rows)
            ])

            det += ((-1) ** j) * self.data[0][j] *  minor.determinanr()
        return det

        
    def inverse_2x2(self):
        if self.shape != (2, 2):
            raise ValueError("inverse_2x2 requires a 2x2 matrix")
        det = self.determinanr()
        if  det == 0:
            raise ValueError("Matrixi i singular, no inverse exists ")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

        
    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0  for  j in  range(n)]
            for i in  range(n)
        ])
    

            

        
A = Matrix([[1,2], [3,4 ]])
B = Matrix([[5,6], [7,8 ]])


print("A + B = ", (A+B).data)
print("A @ B = ", A.matmul(B).data)
print("A.T = ", A.transpose().data)
print("det(A) = ", A.determinanr())
print("A^-1 =" , A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 = ", A.matmul(A.inverse_2x2()).data)











# bai tap
