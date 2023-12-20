import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import Linier_Regresor as LR

data = pd.read_csv('train.csv')
minTetah=-20
maxTetah=20
X = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values  

X = np.c_[np.ones(X.shape[0]), X]


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LR.MyLinearRegression(minTetah,maxTetah)
model.fit(X, y)

plt.scatter(X[:, 1], y, color='blue', label='Data Points')  
plt.plot(X[:, 1], model.predict(X), color='red', label='Regression Line')  
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.title('Linear Regression')
plt.legend()
plt.show()
