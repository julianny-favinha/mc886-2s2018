import numpy as np

class NeuralNetworkOne(object):
    def __init__(self, learningRate, activationFunction, activationDerivativeFunction, inputLayerSize=2, hiddenLayerSize=3, outputLayerSize=1):
        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputLayerSize = outputLayerSize

        self.learning_rate = learningRate

        self.weights1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.weights2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None

        self.activation_function = activationFunction
        self.activation_derivative_function = activationDerivativeFunction


    def forward(self, x):
        self.z2 = np.dot(x, self.weights1)
        self.a2 = self.activation_function(self.z2)
        self.z3 = np.dot(self.a2, self.weights2)
        self.a3 = self.softmax(self.z3)
        return self.a3

    def softmax(self, z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        return e_x / div


    def cost(self, x, y):
        return (np.sum((y - self.a3) ** 2)) / x.shape[0]


    def cost_prime(self, x, y):
        delta3 = np.multiply(-(y - self.a3), self.activation_derivative_function(self.z3))
        djdw2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.weights2.T) * self.activation_derivative_function(self.z2)
        djdw1 = np.dot(x.T, delta2)

        return djdw1, djdw2


    def gradient_descent(self, delta1, delta2):
        self.weights1 -= self.learning_rate * delta1
        self.weights2 -= self.learning_rate * delta2
