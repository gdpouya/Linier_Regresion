import numpy as np
import sys
import matplotlib.pyplot as plt
class MyLinearRegression:
    def __init__(self, minTetah, maxTetah):
        self.minTetah = minTetah
        self.maxTetah = maxTetah
        self.Results = []
        self.tetah0=None
        self.tetah1=None
        self.minloss=2**64 - 1
    
    def cal(self,tetah0,tetah1,X,y):
        samples,features=X.shape
        loss = 0
        for i in range (samples):
            predict=(tetah0 * int(X[i][1])) + tetah1
            loss += pow(predict - int(y[i]), 2)
        self.Results.append((loss, tetah0, tetah1))
        print (f"for {tetah0},{tetah1} loss:{loss}")
        if loss < self.minloss:
            print(f"new value= loss:{loss} t1:{tetah0} t2:{tetah1}")
            self.minloss=loss
            self.tetah0=tetah0
            self.tetah1=tetah1
    
    def fit(self, X, y):
        for i in range(self.minTetah, self.maxTetah):
            for j in range(self.minTetah, self.maxTetah):
                print(f"cal {i} , {j}")
                self.cal(i,j,X,y)
        self.show()
        
    def predict(self,X):
        return [self.tetah0 * xi + self.tetah1 for xi in X]

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def show(self):
        fig = plt.figure()
        M = fig.add_subplot(111, projection='3d')
        theta0_values = np.linspace([theta0 for _,theta0,_ in self.Results], 200)
        theta1_values = np.linspace([theta1 for _,_,theta1 in self.Results], 200)
        loss_values = np.linspace([loss for loss,_,_ in self.Results], 100000000000)
        M.plot_surface(theta0_values, theta1_values, loss_values, cmap='viridis')
        M.set_xlabel("Theta0")
        M.set_ylabel("Theta1")
        M.set_zlabel('loss')
        plt.show()