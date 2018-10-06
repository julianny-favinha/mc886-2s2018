import numpy as np

class NNOL(object):
    def __init__(self, learning_rate, inputLayerSize=2, hiddenLayerSize=3, outputLayerSize=1):
        """Atribuicao de hiperparametros"""
        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputLayerSize = outputLayerSize
        # self.rglzn_lambda = 1
        self.learning_rate = learning_rate
        """Inicializa parametros"""
        self.weights1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.weights2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None
        self.j = None


    def forward(self, x):
        self.z2 = np.dot(x, self.weights1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights2)
        self.a3 = self.sigmoid(self.z3)
        return self.a3


    @staticmethod
    def sigmoid_prime(z):
        sig = 1.0 / (1.0 + np.exp(-z))
        return sig * (1 - sig)


    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))


    def iteration(self, x, y):
        self.a3 = self.forward(x)
        self.cost(x, y)
        self.cost_prime(x, y)
        self.gradient_descent()
        return self.j


    def cost(self, x, y):
        self.j = (np.sum((y - self.a3) ** 2)) / x.shape[0] #/x.shape[0] + (self.rglzn_lambda/2)*((np.sum(self.weights1**2))+(np.sum(self.weights2**2)))


    def cost_prime(self, x, y):
        # delta2 = y - self.a3
        # self.djdw2 = np.dot(self.a2.T, delta2)
        #
        # delta1 = np.dot(delta2, self.weights2.T) * self.sigmoid_prime(self.a2)
        # self.djdw1 = np.dot(x.T, delta1)

        # print(self.djdw1)
        # print(self.djdw2)

        # print("cost_prime")
        delta3 = np.multiply(-(y - self.a3), self.sigmoid_prime(self.z3))
        self.djdw2 = np.dot(self.a2.T, delta3)# /x.shape[0] + self.rglzn_lambda*self.weights2
        delta2 = np.dot(delta3, self.weights2.T) * self.sigmoid_prime(self.z2)
        self.djdw1 = np.dot(x.T, delta2)#/x.shape[0] + self.rglzn_lambda*self.weights1


    def gradient_descent(self):
        self.weights1 -= self.learning_rate * self.djdw1
        self.weights2 -= self.learning_rate * self.djdw2
