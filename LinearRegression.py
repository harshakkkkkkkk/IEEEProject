import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X, y): #For Training => initializing w and b to 0
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weight) + self.bias

            #The derivative / gradient of w and b
            dw = 1 / n_samples * np.dot(X.T, (y_pred - y))
            db = 1 / n_samples * np.sum(y_pred - y)

            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db
        
    def predict(self, X): #For Inference
        y_pred = np.dot(X, self.weight) + self.bias
        return y_pred