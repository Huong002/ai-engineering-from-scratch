def  numerical_derivate(f,x , h=1e-7):
    """Ước lượng đạo hàm bậc nhất của hàm `f` tại `x` bằng sai phân trung tâm.

    Args:
        f: Hàm một biến cần lấy đạo hàm.
        x: Điểm cần ước lượng đạo hàm.
        h: Bước dịch nhỏ dùng để tính gần đúng.

    Returns:
        Giá trị gần đúng của đạo hàm `f` tại `x`.
    """
    
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):   
    """Tính bình phương của một số `x`."""
    # tra   ve x binh phuongg
    return x ** 2

for x  in [-2, -1, 0 , 1 , 2]:
    numerical = numerical_derivate(f,x)
    analytical =  2 * x
    print(f"x={x:2d}  f'(x) numerical = {numerical:.6f}  analytical = {analytical:.1f}")    


def numerical_gradient(f, point, h=1e-7):
    """Ước lượng gradient của hàm nhiều biến `f` tại `point`.

    Args:
        f: Hàm nhận một danh sách tọa độ và trả về một số.
        point: Tọa độ điểm cần tính, ví dụ `[1.0, 2.0]`.
        h: Bước dịch nhỏ dùng để ước lượng từng đạo hàm riêng.

    Returns:
        Danh sách các đạo hàm riêng, theo thứ tự tọa độ trong `point`.
    """
    gradient = []
    for i in  range(len(point)):
        point_plus = list(point) 
        point_minus =  list(point)
        point_plus[i]  +=  h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)

    return gradient

def f_multi(point):
    """Tính `x² + 3xy + y²` với `point` chứa hai tọa độ `[x, y]`."""
    x , y = point
    return x ** 2  + 3 * x * y  + y **  2


grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Analytical gradient at (1,2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]") 


x = 5.0
lr = 0.1
for step in  range(20):
    grad = 2*x 
    x = x  - lr *grad
    print(f" step {step:2d} x={x:8.4f}  f(x)={x**2:10.6f}")

def f_2d(point):
    """Tính tổng bình phương hai tọa độ `[x, y]`: `x² + y²`."""
    x , y = point
    return x**2 + y**2 

point = [4.0,  3.0]
lr = 0.1
for step in range(30):
    grad =  numerical_gradient(f_2d,  point)
    point = [p - lr * g  for p , g in  zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or  step == 29:
        print(f"step {step:2d} point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")

        
import math

test_functions = [
    ("x^2", lambda x: x**2, lambda x: 2*x),
    ("x^3", lambda x: x**3, lambda x: 3*x**2),
    ("sin(x)", lambda x: math.sin(x), lambda x: math.cos(x)),
    ("e^x", lambda x: math.exp(x), lambda x: math.exp(x)),
    ("1/x", lambda x: 1/x, lambda x: -1/x**2)
]
x =  2.0 
print(f"{'Funcion':<12} {'Numerical':>12} {'Error':>12}")
print("-"  *  50 )
for  name , f, df in test_functions:
    num = numerical_derivate(f,x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")

   
def hessian_2d(f,x, y ,  h = 1e-5) :
    """Ước lượng ma trận Hessian của hàm hai biến tại điểm `(x, y)`.

    Args:
        f: Hàm hai biến nhận `x` và `y`.
        x: Tọa độ thứ nhất của điểm cần tính.
        y: Tọa độ thứ hai của điểm cần tính.
        h: Bước dịch nhỏ dùng để ước lượng các đạo hàm bậc hai.

    Returns:
        Ma trận 2×2 gồm các đạo hàm riêng bậc hai.
    """
    fxx = (f(x+h, y)  - 2 * f(x,y)  + f(x-h, y)) / (h**2)
    fyy = (f(x, y +h)  - 2  * f(x,y) + f(x, y -h )) / (h**2)
    fxy = (
        f(x + h, y + h)
        - f(x + h, y - h)
        - f(x - h, y + h)
        + f(x - h, y - h)
    ) / (4 * h**2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x,y):
    """Tính `x² - y²`, hàm có dạng yên ngựa quanh gốc tọa độ."""
    return x ** 2 - y  **  2

def  bowl(x,y):
    """Tính `x² + y²`, tạo bề mặt hình bát quanh gốc tọa độ."""
    return x **2  + y ** 2 

H_saddle = hessian_2d(saddle, 0.0,  0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"Sale Hessian: {H_saddle}")
print(f"Bowl Hessian: {H_bowl}")



import math

def taylor_approx(f, f_prime, f_double_prime,  x0, h, order = 2):
    """Ước lượng `f(x0 + h)` bằng khai triển Taylor đến bậc chỉ định.

    Args:
        f: Hàm cần xấp xỉ.
        f_prime: Đạo hàm bậc nhất của `f`.
        f_double_prime: Đạo hàm bậc hai của `f`.
        x0: Điểm khai triển.
        h: Khoảng cách từ `x0` đến điểm cần xấp xỉ.
        order: Bậc khai triển hỗ trợ: 0, 1 hoặc 2.

    Returns:
        Giá trị xấp xỉ của `f(x0 + h)`.
    """
    result = f(x0)
    if order >= 1:
        result+= f_prime(x0)* h 
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h **  2
        
    return result

    
x0 =  0.0
for h in  [0.1,  0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 =  taylor_approx(math.sin, math.cos, lambda x: -math.sin(x),  x0,  h, order=1)
    t2 =  taylor_approx(math.sin, math.cos, lambda x: -math.sin(x),  x0,  h, order=2)

    print(f"h={h:.1f} sin(h={true_val:.4f} order1={t1:.4f}  order2={t2:.4f})")



import random

random.seed(42)

w = random.gauss(0, 1)
b= random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0 , 4.0, 5.0]
xy = [3.0, 5.0, 7.0 , 9.0, 11.0]

for epoch  in range(1500):

    total_loss = 0 
    dw = 0 
    db = 0 
    for x, y in zip(xs, xy):
        pred = w * x + b
        error= pred -y 
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error

    dw /= len(xs)
    db /= len(xs)

    total_loss /= len(xs)
    w -=  lr * dw
    b -=  lr * db

    if epoch  % 40 == 0 or epoch == 199:
        print(f"epochh {epoch:3d}  w={w:.4f} b={b:.4f} losss={total_loss:.6f}")


print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")
