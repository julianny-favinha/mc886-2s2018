import numpy as np 

def sigmoid(coefficients, xs):
	print("dot",np.dot(coefficients, xs.T))
	return 1 / (1 + np.exp(-np.dot(coefficients, xs.T)))

def loss_function(layer3, y):
	k = layer3.shape[0]

	sum = 0
	for index in range(k):
		sum += np.sum(y[index]*np.log(layer3[index] + 0.00000001) + (1 - y[index])*np.log(1 - layer3[index] + 0.00000001))
	
	return -(1/len(xs)) * sum

def feed_forward(layer1, weights):
	# first hidden layer
	layer2 = sigmoid(weights[0], layer1)
	print("layer2.shape", layer2.shape)
	print(layer2)

	# output
	layer3 = sigmoid(weights[1], layer2)
	print("layer3", layer3)

	return layer3

if __name__ == "__main__":
	# input
	entrada = np.array([[1, 2, 3]])
	
	weights1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	weights2 = np.array([10, 11, 12])
	# cada weight dentro vai ter um shape!
	weights = [weights1, weights2]

	# mesmo shape de layer3
	y = np.array([[1]])

	layer3 = feed_forward(entrada[0], weights)

	loss = loss_function(layer3, y[0])

	# derivative_error()

	# backpropagate()

	# update_weights()