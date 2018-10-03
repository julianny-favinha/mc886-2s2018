import numpy as np

class NeuralNetwork(object):
	def __init__(self):
		self.inputLayerSize = 785
		self.hiddenLayerSize = 256
		self.outputLayerSize = 10

		self.weights1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
		self.weights2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)
		self.bias2 = np.ones((500, self.outputLayerSize))

	def forward(self, X):
		# print("X[0]", X[0])
		# print("weights1[0]", self.weights1[0])
		self.z2 = np.dot(X, self.weights1)
		# print("z2[0]", self.z2[0])
		self.a2 = self.sigmoid(self.z2)
		# print("a2[0]", self.a2[0])
		self.z3 = np.dot(self.a2, self.weights2) #+ self.bias2
		# print("z3[0]", self.z3[0])
		self.a3 = self.softmax(self.z3)
		# print("a3[0]", self.a3[0])
		return self.a3

	def sigmoidDerivative(self, z):
		# print("sigmoidDerivative")
		# print(np.exp(-z)/((1+np.exp(-z))**2))
		return self.sigmoid(z) * (1 - self.sigmoid(z))
		# return np.exp(-z)/(((1+np.exp(-z))**2))

	def sigmoid(self, z):
		# z = np.clip(z, 0.00001, 1)
		z = 1.0 / (1.0 + np.exp(-z))
		return z

	# def softmax(self, z):
	# 	exponent = np.exp(z - np.max(z))
	# 	return exponent / np.sum(exponent, axis=0)

	def softmax(self, z):
		assert len(z.shape) == 2
		s = np.max(z, axis=1)
		s = s[:, np.newaxis]
		e_x = np.exp(z - s)
		div = np.sum(e_x, axis=1)
		div = div[:, np.newaxis]
		return e_x / div

	# def reLU(self, z):
	# 	return np.maximum(np.zeros(z.shape), z)

	# def reLUDerivative(self, z):
	# 	new_z = np.copy(z)
	# 	new_z[new_z > 0] = 1
	# 	new_z[new_z <= 0] = 0
	# 	return new_z

	def CrossEntropy(self, h, y):
		return (-1 / len(y)) * (np.sum(np.multiply(y, np.log(h + 0.00000001))))
		#return np.mean(np.sum((y - h)**2, axis=1))

	def CrossEntropyDerivative(self, X, y):
		delta3 = self.a3 - y
		a2delta3 = np.dot(self.a2.T, delta3)

		delta2 = np.multiply(np.dot(delta3, self.weights2.T), self.sigmoidDerivative(self.z2))
		a1delta2 = np.dot(X.T, delta2)

		#deltabias2 = self.sigmoidDerivative(self.z3)
		# print("deltabias2.shape", deltabias2.shape)

		return a1delta2, a2delta3