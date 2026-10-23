class Vector:
    def __init__(self, components):
        self.components = components
        self.dim = len(self.components)

        
    def __add__(self, other):
        return Vector([a + b for a,b in zip(self.component, other.components)])

        
    def __sub__(self, other):
        return Vector([a -b for a, b in  zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b  for a,b in zip(self.components, other.components))

        
    def mangitude(self):
        return  sum(x **  2 for  x in self.components) ** 0.5
    
    
    def normalize(self):
        mag = self.mangitude()
        return Vector([x / mag for  x in self/self.components])
    
    
    def cosine_similarity(self, other):
        return self.dot(other) / (self.mangitude() * other.mangitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1,2,3])
b = Vector([4,5,6])


print(f" a  + b = { a+b}")
print(f" a  . b = { a.dot(b)}")
print(f" |a|  = {a.mangitude():.4f}")
print(f"cosine ssimileariy =  {a.cosine_similarity(b):.4f}")


class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for  row  in  rows]
        self.shape= (len(self.rows), len(self.rows[0]))

    def  __matmul__(self, other ):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] *  other.components[j] for  j in range(self.shape[1])) 
                for i in range(self.shape[0])
            ])

        rows = []
        
        for i in  range(self.shape[0]):
            row =  []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k  in  range(self.shape[1])
                ))

                rows.append(row)

            return Matrix(rows)
            
    def  transpose(self):
        """Bien  hang thanh cot , cot thanh hang -> tra ve ma tran moi- ma tran goc van ko bi sua (Ma tran m x n ->  n x m )
        """
        return Matrix([
           [self.rows[j][i] for j in range(self.shape[0])] 
           for i  in range(self.shape[1])
        ])
        
    def __repr__(self):
        return f"Matrix{self.rows}"

rotation_90 = Matrix([[0,-1], [1, 0]])
point = Vector([3,1])


rotated =  rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90: {rotated}")

import random

random.seed(2)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector =  Vector([1.0 , 0.5, -0.3])

output =  weights @ input_vector
print(f"input (3D): {input_vector}")
print(f"output (2D): {output}")


# doc lap  tuyen tinh va pep chieu  ttu  dau
def is_linearly_independents(vectors):

    n = len(vectors)
    dim  = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:]  for row in mat.rows]
    rank = 0 
    
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break

        if  pivot  is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale =   rows[rank][col]
        rows[rank]  = [x/scale for x in rows[rank]]
        
        for row in range(len(rows)):
            if row  !=  rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j]  - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank  ==  n

    
def project(a,b):
    scalar = a.dot(b) /  b.dot(b)
    return Vector([scalar * x for x in  b.componennts])

    
        
        

    
