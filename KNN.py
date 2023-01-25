
import numpy as np
import math
from keras.datasets import mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()

class KNN:
    def __init__(self, train_x, train_y, K=3):
        self.K = K
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.N = len(train_x)

    def pred(self, X):
        X = np.array(X)

        # Calculate eucledian distance from each point
        dist = [math.sqrt(np.sum(np.square(train - X))) for train in self.train_x]

        # sort arrays
        sorted = np.argsort(dist)

        results = np.zeros(10)

        # find most used values
        for i in range(self.K):
            index = sorted[i]
            results[self.train_y[index]] += 1
        
        return np.argmax(results)


# find error rate on testing set
knn = KNN(train_x[:1000], train_y[:1000], 10)

errors = 0

for i in range(1000):
    x = test_x[i]
    y = test_y[i]
    prediction = knn.pred(x)
    if  prediction != y:
        errors += 1

print("error rate: " + str(errors / 1000))
        








