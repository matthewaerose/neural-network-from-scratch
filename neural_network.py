"""
Building a neural network from scratch will help me understand what is really going on.
using this as a reference/guide:
        https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
"""

"""
Training a NN -> process of fine-tuning the weights and biases from the input data

feedforward -> calculating the predicted output

backpropagation -> updating the weights and biases 

yHat = sigma(W_2sigma(W_1x +b1 ) + b2)

Loss function evaluates the "goodness of our predictions. Nature of the problem we are trying to solve should 
dictate our choice of function. Here, we will use a simple sum of squares error. 
SUM(y - yHat)^2

Goal in training is to find the best set of weights and biases that minimizes the loss function


"""
from scipy.special import expit
import numpy as np
import math
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, x, y):
        """

        :param x:
        :param y:
        """
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], 4)
        self.weights2   = np.random.rand(4, 1)
        self.y          = y
        self.output     = np.zeros(y.shape)
        self.output_0 = []
        self.output_1 = []
        self.output_2 = []
        self.output_3 = []

    def feedforward(self):
        """

        :return:
        """
        self.layer1 = expit(np.dot(self.input, self.weights1))
        self.output = expit(np.dot(self.layer1, self.weights2))
        self.output_0.append(self.output[0])
        self.output_1.append(self.output[1])
        self.output_2.append(self.output[2])
        self.output_3.append(self.output[3])
    def backpropagation(self):
        """
        application of the chain rule to find derivative of the loss function with respect to W1 & W2
        :return:
        """
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
        a1 = self.y - self.output
        a2 = self.sigmoid_derivative(self.output)
        a = np.dot(2 * a1 * a2, self.weights2.T)
        d_weights1 = np.dot(self.input.T, a * self.sigmoid_derivative(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def sigmoid(self, x):
        """
        diy sigmoid which will serve as our activation function
        :param x:
        :return:
        """
        var = math.exp(x)
        bottom = var + 1
        return var / bottom

    def sigmoid_derivative(self, x):
        """

        :param x:
        :return:
        """
        left = expit(x)
        right = 1 - left
        return left * right

    def start(self, iterations):
        for i in range(iterations):
            if i % 100 == 0:
                print(f"Iteration: {i}")
            self.feedforward()
            self.backpropagation()


# Pro tip: how the arrays are input matters a great deal...
# at first tried entering as a 3x4 instead of a 4x3
x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0],
])

nn = NeuralNetwork(x, y)

nn.start(15000)

plt.plot(nn.output_0)
plt.plot(nn.output_1)
plt.plot(nn.output_2)
plt.plot(nn.output_3)

plt.legend(['actual 0', 'actual 1', 'actual 1', 'actual 0'])
plt.ylabel("loss")

print(nn.output)

plt.show()

"""
Lessons learned: use built in functions, instead of trying to make your own. Trying to program a sigmoid function for
N dimension arrays isn't fun. 
"""
