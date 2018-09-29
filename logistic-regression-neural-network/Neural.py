import numpy as np 

def sigmoid(coefficients, xs):
	print("dot",np.dot(coefficients, xs.T))
	return 1 / (1 + np.exp(-np.dot(coefficients, xs.T)))


if __name__ == "__main__":
	layer1 = np.array([1, 2, 3])
	weights1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

	weights2 = np.array([10, 11, 12])
	print("weights2.shape",weights2.shape)

	layer2 = sigmoid(weights1, layer1)
	print("layer2.shape", layer2.shape)
	print(layer2)

	layer3 = sigmoid(weights2, layer2)
	print("layer3", layer3)