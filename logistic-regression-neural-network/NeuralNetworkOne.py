import numpy as np

# def reLU(z):
# 	return np.maximum(np.zeros(z.shape), z)

class NeuralNetwork(object):
	def __init__(self):
		self.inputLayerSize = 3
		self.outputLayerSize = 2
		self.hiddenLayerSize = 3

		self.weights1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
		self.weights2 = np.array([[10, 11], [12, 13], [14, 15]])

	def forward(self, X):
		self.z2 = np.dot(X, self.weights1)
		self.a2 = self.(self.z2)
		self.z3 = np.dot(self.a2, self.weights2)
		self.outputLayer = self.sigmoid(self.z3)
		return self.outputLayer

	def sigmoidPrime(self, z):
		return np.exp(-z)/((1+np.exp(-z))**2)

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def lossFunction(self, h, y):
		sum = np.sum(y*np.log(h + 0.00000001) + (1 - y)*np.log(1 - h + 0.00000001))
		return -(1/y.shape[0]) * sum

	def J(self, h, y):
		return np.sum(0.5 * (y - h)**2)

	def JPrime(self, X, y):
		self.output_layer = self.forward(X)
		
		delta3 = np.multiply(-(y - self.output_layer), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)

		delta2 = np.dot(delta3, self.weights2.T) * self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)

		return dJdW1, dJdW2