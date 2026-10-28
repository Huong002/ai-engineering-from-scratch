def  numerical_derivative(f, x, h=1e-7):
    return (f(x+h) - f(x-h))/ (2*h)

    
def f(x):
    return x ** 2



for x in [-2, -1, 0 ,1 , 2]:
    numerical  = numerical_derivative(f,  x)
    analycical =  2* x
    print(f"x =  {x:2d}  f'(x) numerical = {numerical:.6f}  analycial = {analycical:.1f}")

    
    
    