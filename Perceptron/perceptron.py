# import libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# generate data
def generate_data(NumofSamples):
    X = np.zeros(shape=(NumofSamples, 2))
    y = np.zeros(NumofSamples)
    for i in range(NumofSamples):
        X[i, 0] = random.randint(0,20)
        X[i, 1] = random.randint(0,20)
        if (X[i,0] + X[i, 1]) > 20:
            y[i] = 1
        else:
            y[i] = -1

    return X, y

class Person():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def myfunc(self):
        print("Hello, my name is : {}".format(self.name))

class Perceptron():
    def __init__(self, b=0, max_iter=10000):
        # we initialize an instance
        self.max_iter = max_iter
        self.w = []
        self.b = 0
        self.num_samples = 0
        self.num_features = 0

    def train(self, X, y):
        # set the number of samples and features
        self.num_samples, self.num_features = np.shape(X)
        # set weight factor
        self.w = np.zeros(self.num_features)

        # run max_iter 's turn
        for i in range(self.max_iter):
            # weight does not need to update (initialize)
            w_update = False
            # inner loop to check are there any misclassified example
            for j in range(self.num_samples):
                output = np.dot(X[j], self.w) + self.b
                # if the model misclassified
                if y[j] * output <= 0:
                    w_update = True
                    # use perceptron algorithms
                    self.w += y[j] * X[j]
                    self.b += y[j]

            if (not w_update):
                print("Convergence reached in %i iterations." %i)
                break

        if (w_update):
            print(
            """
            WARNING: convergence not reached in %i iterations.
            Either dataset is not linearly separable, 
            or max_iter should be increased
            """ % self.max_iter
                )

    def classify_element(self, x_elem):
        return np.sign(np.dot(x_elem, self.w) + self.b)

    def classify(self, X):
        out = np.dot(X, self.w)
        predicted_Y = np.sign(out + self.b)
        return predicted_Y

# experiment part

X, y = generate_data(200)

x_pos = []
# TO Do: Insert your code to find the indices for negative examples
x_neg = []

for i in range(X.shape[0]):
    if (y[i]) == 1:
        x_pos.append(i)
    else:
        x_neg.append(i)
# make a scatter plot
plt.scatter(X[x_pos, 0], X[x_pos, 1], color='Blue')
plt.scatter(X[x_neg, 0], X[x_neg, 1], color='Red')
plt.show()

p = Perceptron()

p.train(X, y)

predicted_y  = p.classify(X)

acc = accuracy_score(predicted_y, y)
print(acc)


x1 = np.arange(0, 20, 0.1)
# bias
b = p.b
# weight vector
w = p.w
# we now use list comprehension to generate the array of the second feature
# To do: generate the second features for the hyperplane, i.e., (X1[i], X2[i]) is an point in the hyperplane
x2 = [(-b-w[0]*x)/w[1] for x in x1]
plt.scatter(X[x_pos, 0], X[x_pos, 1], color='blue')
plt.scatter(X[x_neg, 0], X[x_neg, 1], color='red')
# plot the hyperplane corresponding to the perceptron
plt.plot(x1, x2, color="black", linewidth=2.5, linestyle="-")
plt.show()