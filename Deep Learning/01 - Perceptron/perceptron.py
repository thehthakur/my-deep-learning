import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def net_input(self, x):
        return np.dot(x, self.w_) + self.b_

    def predict(self, x):
        return np.where(self.net_input(x) >= 0, 1, 0)

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)       
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0)
        self.errors_ = []
        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self

# Example Usage
perceptron = Perceptron(eta=0.01, n_iter=10)

X = np.array([[1, -2, 0, -1], [0, 1.5, -0.5, -1], [-1, 1, 0.5, -1]])
y = np.array([-1, -1, 1])

perceptron.fit(X, y)

print("Weights:", perceptron.w_)
print("Bias:", perceptron.b_)
print("Errors in each epoch:", perceptron.errors_)

# Predicting with new data
new_data = np.array([0.5, -1, 0.3, -0.5])
prediction = perceptron.predict(new_data)
print("Prediction for new data:", prediction)
