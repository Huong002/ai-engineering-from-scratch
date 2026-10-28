
import numpy as  np 

class NearestCentroid:
    def fit(self, X, y ):
        """Ham nay  tyinh toan hai  gia tri  trung  binh"""
        self.classes=  np.unique(y)
        self.centroids = np.array([
            X[y == c].mean(axis  = 0 ) for c in self.classes
        ])

        
    def  predict(self, X):
        """Tinh toan khhooang cach"""
        distances =  np.array([
            np.sqrt(((X -c) **  2).sum(axis =  1))
            for c in self.centroids
        ])
        return self.classes[distances.argmin(axis=0)]

    
# X_train  = np.array([
#     [1,2],
#     [2,3],
#     [2,1],
#     [8,9],
#     [9,8],
#     [8,8]
# ])

# y_train  = np.array([0,0,0,1,1,1])


# X_test = np.array([
#     [2,2],
#     [9,9],
#     [5,5],
# ])

# # huan luyen  moo hinh 
# model =  NearestCentroid()
# model.fit(X_train, y_train)

# # du  doan 
# predictions = model.predict(X_test)

# print("Cac lop: ", model.classes)
# print("Tam  cum: ", model.centroids)
# print("Du lieu kiem tra: \n", X_test)
# print("Ket qua du doan: ", predictions)


from  sklearn.neighbors import NearestCentroid
from  sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=500, n_features=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42
)

X_train ,  X_test, y_train , y_test = train_test_split(X,y,test_size=0.3)


clf = NearestCentroid()
clf.fit(X_train, y_train)
print(f"Do chinh xac: {clf.score(X_test, y_test):.3f}")

