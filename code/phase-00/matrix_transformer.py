import math


def rotation_2d(theta):
    """Xoay diem  quanh goc toa do  

    Args:
        theta (_type_): goc toa do  (tinh bang radian )

    """
    c,s  = math.cos(theta), math.sin(theta)
    return[[c,-s], [s,c]]


def scaling_2d(sx,  sy ):
    """Nhan toa do x voi  sx,  t voi  sy

    """
    return [[sx,0], [0,sy]]


def shearing_2d(kx, ky):
    """Tao phep  bien  doi  lam nghien hinh"""
    return [[1,kx], [ky, 1]]

    
def reflection_x():
    """Phan chieu qua truc  x"""
    return [[1, 0], [0,-1] ]

def reflection_y():
    """Phan chieu qua  truc  y  """
    return [[-1,0],  [0,1]]

def mat_vec_mul(matrix, vector):

    """Ham nhan ma ttran voi vector  diem  roi tra ve vector moi  

    Returns:
        ham tra  ve vector moi. moi phan tu  la tong cua mot hang torng  matrix nhhan  lan luot cac phan  tu tuong ung terng vecto
    """
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in  range(len(matrix))
    ]

def mat_mul(a,b):
    """Nhan 2  ma tran
    Returns:
        cott cuua a phai bag  so hang cua b
        ket qua  co soo hang bang a va so cot bang b
    """
    rows_a,  cols_b  = len(a),  len(b[0])
    cols_a = len(a[0])

    return [
        [
            sum(a[i][k] * b[k][j] for k in range(cols_a))
            for j in range(cols_b)
        ]
        for i in range(rows_a)
    ]
    
    
    
point =  [1.0,  0.0] # vetor 
angle =  math.pi / 4   # goc: 45 do  


rotated =  mat_vec_mul(rotation_2d(angle),  point)
print(f"Rotate (1,0) by 45 deg: ({rotated[0]:.4f}, {rotated[1]:.4f})")

scaled = mat_vec_mul(scaling_2d(2,3), [1.0, 1.0])
print(f"Scale (1,1) by (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")

sheared  = mat_vec_mul(shearing_2d(1,0), [1.0,  1.0])
print(f"Shear (1,1) kx = 1: ({sheared[0]:.1f}, {sheared[1]:.1f})")


reflected = mat_vec_mul(reflection_y(),  [2.0, 1.0])
print(f"Reflect (2,1) aross y: ({reflected[0]:.1f},  {reflected[1]:.1f})")



# Hop thanhh  ccac pkhehp boien  doi  

R =rotation_2d(math.pi /2 )
S = scaling_2d(2, 0.5)

rotated_then_scale = mat_mul(S, R)
scale_then_rotate = mat_mul(R,S)

point = [1.0, 0.0]
result1 =  mat_vec_mul(rotated_then_scale, point)
result2 =  mat_vec_mul(scale_then_rotate, point)


print(f"Rotate 90 them scale: ({result1[0]:.2f}, {result1[1]:.2f})")
print(f"scale then rotate 90: ({result2[0]:.2f}, {result2[1]:.2f})")
print(f"Same?  {result1 == result2}")



"""
Tri rieng 
A @ v = λ * v
"""

def eigenvalues_2x2(matrix):
    a , b =  matrix[0]
    c , d  =  matrix[1]

    trace = a+ d 
    det = a *d  - b* c
    discrimiant = trace **  2  - 4 * det
    if discrimiant <  0:
        real = trace / 2
        imag = (-discrimiant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqtr_disc = discrimiant ** 0.5
    
    return ((trace + sqtr_disc )/ 2 , (trace - sqtr_disc)/ 2)

def eigenvalues_2x2(matrix, eigenvalue):
    a , b =  matrix[0]
    c , d  =  matrix[1]

    if  abs(b) > 1e-10:
        v =  [b, eigenvalue -a ]
    elif abs(c) > 1e-10:
        v = [eigenvalue -d , c] 
    else:
        if abs(a- eigenvalue) > 1e-10:
            v= [1,0]
        else:
            v = [0,1]

    mag = (v[0] ** 2 + v[1] **2) ** 0.5
    return [v[0] /mag, v[1] /mag]


A = [[2, 1], [1, 2]]
vals = eigenvalues_2x2(A)
print(f"Matrix: {A}")
print(f"Eigenvalues: {vals[0]:.4f}, {vals[1]:.4f}")

for val in vals:
    vec = eigenvalues_2x2(A, val)
    result = mat_vec_mul(A, vec)
    scaled = [val * vec[0], val * vec[1]]
    print(f"  lambda={val:.1f}, v={[round(x,4) for x in vec]}")
    print(f"    A@v = {[round(x,4) for x in result]}")
    print(f"    l*v = {[round(x,4) for x in scaled]}")
    
    
   
   
   
"""
DINH THUCc:  he so co gian the tich 
"""

def det_2x2(matrix):
    return matrix[0][0]  *  matrix[1][1] - matrix[0][1] * matrix[1][0]

print(f"det(rotation 45) = {det_2x2(rotation_2d(math.pi/4)):.4f}")
print(f"det(scale 2,3)   = {det_2x2(scaling_2d(2, 3)):.1f}")
print(f"det(shear kx=1)  = {det_2x2(shearing_2d(1, 0)):.1f}")
print(f"det(reflect y)   = {det_2x2(reflection_y()):.1f}")

singular = [[1,2], [2,4]]
print(f"det(sinfgular)   =  {det_2x2(singular):.1f}")
print("Singular: colums  are proiporional, space collapses to a line ")

import numpy as np
theta = np.pi / 4
R = np.array([
    [np.cos(theta),  -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

point = np.array([1.0, 0.0 ])
print(f"Rotate  (1,0) by 45 deg: {R@point}")

S = np.diag([2.0, 3.0])
composed = S @ R
print(f"Scale(2,3) after Rotate(45): {composed @ point}")

A = np.array([[2, 1], [1, 2]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"  A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

print(f"\ndet(R) = {np.linalg.det(R):.4f}")
print(f"det(S) = {np.linalg.det(S):.1f}")

B = np.array([[3, 1], [0, 2]], dtype=float)
vals, vecs = np.linalg.eig(B)
D = np.diag(vals)
V = vecs
reconstructed = V @ D @ np.linalg.inv(V)
print(f"\nEigendecomposition A = V @ D @ V^-1:")
print(f"Original:\n{B}")
print(f"Reconstructed:\n{reconstructed}")

def rotation_3d_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotation_3d_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

point_3d = np.array([1.0, 0.0, 0.0])
rotated_z = rotation_3d_z(np.pi / 2) @ point_3d
rotated_x = rotation_3d_x(np.pi / 2) @ point_3d

print(f"\n3D point: {point_3d}")
print(f"Rotate 90 around z: {np.round(rotated_z, 4)}")
print(f"Rotate 90 around x: {np.round(rotated_x, 4)}")