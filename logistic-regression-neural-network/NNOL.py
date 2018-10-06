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
        self.weights1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize) / np.sqrt(self.inputLayerSize)
        self.weights2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize) / np.sqrt(self.hiddenLayerSize)
        self.bias1 = np.zeros((1, self.hiddenLayerSize))
        self.bias2 = np.zeros((1, self.outputLayerSize))
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None
        self.j = None


    def forward(self, x):
        self.z2 = np.dot(x, self.weights1) + self.bias1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights2) + self.bias2
        self.a3 = self.softmax(self.z3)
        return self.a3


    @staticmethod
    def sigmoid_prime(z):
        sig =  1.0 / (1.0 + np.exp(-z))
        return z * (1 - z)


    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))


    @staticmethod
    def softmax(z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        return e_x / div

    def iteration(self, x, y):
        self.a3 = self.forward(x)
        self.cost(y)
        self.cost_prime(x, y)
        self.gradient_descent()
        return self.j


    def cost(self, y):
        # self.j = (np.sum((y - self.a3) ** 2)) / x.shape[0] #/x.shape[0] + (self.rglzn_lambda/2)*((np.sum(self.weights1**2))+(np.sum(self.weights2**2)))
        m = self.a3.shape[0]
        
        self.j = (-1.0 / m) * np.sum(np.sum(y * np.log(self.a3 + 0.0001) + (1.0 - y) * np.log(1.0 - self.a3 + 0.0001), axis=1))


    def cost_prime(self, x, y):
        delta2 = y - self.a3
        self.djdw2 = np.dot(self.a2.T, delta2)
        self.djdb2 = np.sum(delta2, axis = 0, keepdims = True)

        delta1 = np.dot(delta2, self.weights2.T) * self.sigmoid_prime(self.a2)
        self.djdw1 = np.dot(x.T, delta1)
        self.djdb1 = np.sum(delta1, axis = 0)

        # print("cost_prime")
        # delta3 = np.multiply(-(y - self.a3), self.sigmoid_prime(self.z3))
        # self.djdw2 = np.dot(self.a2.T, delta3)# /x.shape[0] + self.rglzn_lambda*self.weights2
        # delta2 = np.dot(delta3, self.weights2.T) * self.sigmoid_prime(self.z2)
        # self.djdw1 = np.dot(x.T, delta2)#/x.shape[0] + self.rglzn_lambda*self.weights1


    def gradient_descent(self):
        self.weights1 -= self.learning_rate * self.djdw1 * 0.01
        self.bias1 -= self.learning_rate * self.djdb1
        self.weights2 -= self.learning_rate * self.djdw2 * 0.01
        self.bias2 -= self.learning_rate * self.djdb2 
