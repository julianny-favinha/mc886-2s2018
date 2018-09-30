import numpy as np



class NeuralNetwork(object):
	def __init__(self):
		self.inputLayerSize = 784
		self.hiddenLayerSize = 784
		self.outputLayerSize = 10

		self.weights1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
		self.weights2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)

	def forward(self, X):
		self.z2 = np.dot(X, self.weights1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.weights2)
		self.outputLayer = self.sigmoid(self.z3)
		return self.outputLayer

	def sigmoidPrime(self, z):
		print("sigmoidPrime")
		print(np.exp(-z)/((1+np.exp(-z))**2))
		return np.exp(-z)/((1+np.exp(-z))**2)

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	# def reLU(self, z):
	# 	return np.maximum(np.zeros(z.shape), z)

	# def reLUPrime(self, z):
	# 	new_z = np.copy(z)
	# 	new_z[new_z > 0] = 1
	# 	new_z[new_z <= 0] = 0
	# 	return new_z

	def lossFunction(self, h, y):
		print("h", h)
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