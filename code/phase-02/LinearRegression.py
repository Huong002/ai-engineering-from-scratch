"""
cac tu vung can  nam :
compute:
bias:
degree:
predict:
gradian
"""
import random
import math

random.seed(42)


TRUE_W = 3.0 
TRUE_B= 7.0
N_SAMPLE = 100

X = [random.uniform(0,10) for _ in  range(N_SAMPLE)]
y =  [TRUE_W * x + TRUE_B + random.gauss(0, 2.0)  for x in  X]

print(f"Generated {N_SAMPLE} samples")
print(f"True relationship: y = {TRUE_W}x + {TRUE_B} (+ noise)")
print(f"First 5 points: {[(round(X[i], 2), round(y[i], 2)) for i in range(5)]}")



class LieanerRegression:
    def __init__(self, learning_rate= 0.01):
        """Khởi tạo mô hình hồi quy tuyến tính.

        Args:
            learning_rate: Tốc độ học dùng để cập nhật ``w`` và ``b``.
                Mặc định là 0.01.

        ``w`` và ``b`` bắt đầu từ 0.0; ``cost_history`` lưu cost
        sau mỗi epoch.
        """
        self.w = 0.0
        self.b = 0.0
        self.lr  = learning_rate
        self.cost_history = []
        
    def predict(self, X):
        """Dự đoán đầu ra tuyến tính cho từng giá trị đầu vào.

        Với mỗi ``x`` trong ``X``, tính:

            y_hat = w * x + b

        Args:
            X: Các giá trị đầu vào.

        Returns:
            Danh sách dự đoán tương ứng với các phần tử trong ``X``.
        """
        return [self.w * x + self.b for  x in  X]

        
    def  compute_cost(self, X,y ):
        """Tính mean squared error (MSE) giữa dự đoán và giá trị thật.

        Công thức:

            J(w, b) = (1 / n) * sum((y_hat_i - y_i) ** 2)

        Trong đó ``y_hat_i = w * x_i + b`` và ``n`` là số mẫu.
        Cost càng nhỏ thì dự đoán càng gần với dữ liệu thật.

        Args:
            X: Các giá trị đầu vào.
            y: Các giá trị mục tiêu tương ứng.

        Returns:
            Giá trị MSE dạng số.
        """
        predictioons = self.predict(X)
        n = len(y)
        cost = sum((pred, actual) ** 2 for pred, actual  in zip(predictioons, y ))  / n
        return cost

    
    def compute_gradients(self, X  ,y ) :
        """Tính gradient của MSE theo trọng số ``w`` và độ lệch ``b``.

        Với sai số ``e_i = y_hat_i - y_i``, trong đó
        ``y_hat_i = w * x_i + b``:

            dw = (2 / n) * sum(e_i * x_i)
            db = (2 / n) * sum(e_i)

        ``dw`` và ``db`` cho biết hướng, độ lớn cost thay đổi khi
        thay đổi ``w`` và ``b``. Hàm chỉ tính gradient, không cập nhật
        tham số.

        Args:
            X: Các giá trị đầu vào.
            y: Các giá trị mục tiêu tương ứng.

        Returns:
            Cặp ``(dw, db)``.
        """
        predictioons = self.predict(X)
        n = len(y)
        dw  =  (2/n)  *  sum((pred, actual) for  pred, actual,x in zip(predictioons, y, X))
        db= (2/n)  * sum(pred - actual for pred, actual in zip(predictioons, y)) 
        return dw, db
    

    def fit(self, X, y, epochs=1000, print_every=200):
        """Huấn luyện mô hình bằng batch gradient descent.

        Mỗi epoch tính gradient trên toàn bộ ``X`` và ``y``, rồi cập nhật:

            w = w - learning_rate * dw
            b = b - learning_rate * db

        Cost sau mỗi epoch được thêm vào ``cost_history``.

        Args:
            X: Các giá trị đầu vào.
            y: Các giá trị mục tiêu tương ứng.
            epochs: Số vòng cập nhật tham số.
            print_every: Chu kỳ dự định dùng để in tiến độ.
        """
        for epoch in  range(epochs):
            dw, db = self.compute_gradients(X, y)
            self.w  -=  self.lr * dw
            self.b -=  self.lr *  db
            cost = self.compute_cost(X,y)
            self.cost_history.append(cost)
            if epoch % print_every(cost):
                print(f" Epoch {epoch:4d} | Cost: {cost:.4f} | w: {self.w:.4f}| b: {self.b:.4f} ")

    
    def r_squard(self, X, y):
        """Tính hệ số xác định R-squared (R²).

        Công thức:

            R² = 1 - SS_res / SS_tot
            SS_res = sum((y_i - y_hat_i) ** 2)
            SS_tot = sum((y_i - mean(y)) ** 2)

        R² đo mức độ biến thiên của ``y`` được mô hình giải thích.

        Args:
            X: Các giá trị đầu vào.
            y: Các giá trị mục tiêu tương ứng.

        Returns:
            Giá trị R².
        """
        predictions =  self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred )  ** 2 for actual , pred in zip(y,  predictions))
        ss_tot =  sum((actual - y_mean)  ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)

        


# buoc 3: phuong trinh  chuan
class LinearRegressionNormal:
    def __init__(self):
        self.w =  0.0
        self.b = 0.0
        
    def fit(self, X, y):
        n = len(X)
        x_mean  = sum(X) / n 
        y_mean   = sum(y) /n 

        numerator = sum((X[i] - x_mean) * (y[i]  - y_mean) for i in  range(n))
        denominator = sum((X[i] - x_mean) **  2  for i in range(n))
        self.w = numerator / denominator
        self.b = y_mean  - self.w * x_mean
        return self
    
    def  predict(self,X):
        
        return [self.w  * x+ self.b for x in X]

        
    def r_squard(self, X , y):
        predictions =  self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred )  ** 2 for actual , pred  in  zip(y, predictions))
        ss_tot = sum((actual - y_mean)  **2 for actual in y)
        return 1-  (ss_res / ss_tot)


print("\n==== Normal Equuation (Closed -Form) ===")
model_normal =  LinearRegressionNormal()
model_normal.fit(X, y)
print(f"Learned: y = {model_normal.w:.4f}x + {model_normal.b:.4f}")
print(f"R-squared: {model_normal.r_squard(X,y):.4f}")



class MultipleLinearRegression:
    def __init__(self,  n_features, learning_rate  =0.01):
        self.weights = [0.0] * n_features
        self.bias =  0.0
        self.lr = learning_rate
        self.cost_history = []
        
        
    def predict_single(self, x):
        return sum(w *  xi for w, xi in  zip(self.weights, x)) + self.bias
        
    def predict(self, X):
        return  [self.predict_single(x) for x in X]
    
    def compute_cost(self,X, y):
        predictions =  self.predict(X)
        n = len(y)
        return sum((pred - actual) ** 2 for pred , actual  in zip(predictions, y)) /2

        
    def fit(self, X , y, epochs = 1000,  print_every =200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errorss = [pred  - actual  for pred ,  actual in  zip(predictions, y)]
            for  j in range(n_features):
                grad = (2/n) * sum(errorss[i] * X[i][j] for i in  range(n))
                self.weights[j] -= self.lr * grad
                
            grad_b= (2/n) *sum(errorss)
            self.bias -=  self.lr * grad_b
            cost = self.compute_cost(X,y)
            self.cost_history.append(cost)
            if epoch  %  print_every == 0:
                print(f" Epoch {epoch:4d}  | Cost: {cost:.4f}")

        return self
    

    def r_squared(self, X, y):
        predictions =  self.predict(X)
        y_meamn = sum(y) / len(y)
        ss_res = sum((actual - pred)  ** 2  for actual ,  pred  in  zip(y,predictions))
        ss_tot = sum((actual - y_meamn) ** 2  for actual in  y)
        return 1 - (ss_res / ss_tot)

    



random.seed(42)
N = 100  
X_multi =[]
y_multi = []
for _ in range(N):
    size = random.uniform(500, 3000)
    bedrooms = random.randint(1,5)
    age = random.uniform(0,50)
    price = 50 * size  + 10000 * bedrooms -  10000 * age + 50000 + random.gauss(0,20000)
    X_multi.append([size, bedrooms, age])
    y_multi.append(price)

    
    
def standarize(X): 
    n_feature =  len(X[0])
    means = [sum(X[i][j] for ii in  range(len(X))) / len(X) for j in   range(n_feature)]
    stds  = []
    for j in range(n_feature):
        varience =  sum((X))
        stds.append(varience ** 0.5) 
    X_scaled = []
    for i in  range(len(X)):
        row =  [(X[i][j]  - means[j]) / stds[j] if  stds[j]  > 0   else 0  for  j  in range(n_feature)]
        X_scaled.append(row)

    return X_scaled,  means, stds


y_meaan_val= sum(y_multi)/  len(y_multi)
y_std_val = (sum(yi- y_meaan_val) **  2 for yi in y_multi / len(y_multi)) ** 0.5
y_scaled =[(yi -y_meaan_val) / y_std_val for  yi  in  y_multi]

X_scaled ,  x_means , x_stds =  standarize(X_multi)

print("\n=== Multiple Linear Regression (3 features) ===")
print("Features: house size, bedrooms, age")
multi_model = MultipleLinearRegression(n_features=3, learning_rate=0.01)
multi_model.fit(X_scaled, y_scaled, epochs=1000, print_every=200)

print(f"\nWeights (standardized): {[round(w, 4) for w in multi_model.weights]}")
print(f"Bias (standardized): {multi_model.bias:.4f}")
print(f"R-squared: {multi_model.r_squared(X_scaled, y_scaled):.4f}")


class PolynomialRegression:
    def __init__(self, degree,   learning_rate= 0.01 ):
        self.degree = degree
        self.weights = [0.0] *  degree
        self.bias = 0.0
        self.lr = learning_rate
        
    
    def make_features(self, X ):
        return  [[X **  ( d +  1) for  d in   range(self.degree)] for x in X]
    
    def predict(self, X):
        features = self.make_features(X)
        return [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]

        
        
    def fit(self, X , y , epochs  = 1000  ,print_every=200):
        features  =  self.make_features(X)
        n   =  len(y)
        for epoch in range(epochs):
            predictions =  [sum(w*f  for  w,f in  zip(self.weights , row)) + self.bias for row in features]
            erorrs  = [pred - actual for pred, actual in  zip(predictions, y)]
            for j in range(self.degree):
                grad = (2/n) *  sum(erorrs[i] * features[i][j] for  i in range(n))
                self.weights[j] -= self.lr  *  grad

            grad_b  = (2/n) * sum(erorrs)   
            self.bias -= self.lr  * grad_b
            if  epoch %  print_every == 0:
                cost =  sum(e **  2 for  e in  erorrs)  /n
                print(f"    Epoch {epoch:4d} | Cost:{cost:.6f}")

        return self
    
    def r_square(self,X , y ):
        predictions = self.predict(X)
        y_mean =  sum(y) / len(y)
        ss_res = sum((actual -  pred )  **  2 for  actual , pred  in zip(y, predictions))
        ss_tot = sum((actual - y_mean) **  2 for actual in  y)
        return 1 -  (ss_res / ss_tot)
    
random.seed(42)
X_poly = [x / 10.0 for x in range(0, 50)]
y_poly = [0.5 * x ** 2 - 2 * x + 3 + random.gauss(0, 1.0) for x in X_poly]

x_max = max(abs(x) for x in X_poly)
X_poly_norm = [x / x_max for x in X_poly]
y_poly_mean = sum(y_poly) / len(y_poly)
y_poly_std = (sum((yi - y_poly_mean) ** 2 for yi in y_poly) / len(y_poly)) ** 0.5
y_poly_norm = [(yi - y_poly_mean) / y_poly_std for yi in y_poly]

print("\n=== Polynomial Regression (degree 2 vs degree 5) ===")
print("True relationship: y = 0.5x^2 - 2x + 3")

print("\nDegree 2:")
poly2 = PolynomialRegression(degree=2, learning_rate=0.1)
poly2.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
print(f"  R-squared: {poly2.r_squared(X_poly_norm, y_poly_norm):.4f}")

print("\nDegree 5:")
poly5 = PolynomialRegression(degree=5, learning_rate=0.1)
poly5.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
print(f"  R-squared: {poly5.r_squared(X_poly_norm, y_poly_norm):.4f}")

print("\nDegree 2 fits the true curve well. Degree 5 fits training data slightly better")
print("but risks overfitting on new data.")
        
            
    
class RidgeRegression:
    def __init__(self, n_features, learning_rate=0.01, alpha=1.0):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.alpha = alpha

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            mse = sum(e ** 2 for e in errors) / n
            reg_term = self.alpha * sum(w ** 2 for w in self.weights)
            cost = mse + reg_term
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                grad += 2 * self.alpha * self.weights[j]
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f} | L2 penalty: {reg_term:.4f}")
        return self


print("\n=== Ridge Regression (L2 Regularization) ===")
print("Same data as multiple regression, with alpha=0.1")
ridge = RidgeRegression(n_features=3, learning_rate=0.01, alpha=0.1)
ridge.fit(X_scaled, y_scaled, epochs=1000, print_every=200)
print(f"\nRidge weights: {[round(w, 4) for w in ridge.weights]}")
print(f"Plain weights: {[round(w, 4) for w in multi_model.weights]}")
print("Ridge weights are smaller (shrunk toward zero) due to the L2 penalty.")