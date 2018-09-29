import numpy as np 

def sigmoid(coefficients, xs):
	print("dot",np.dot(coefficients, xs.T))
	return 1 / (1 + np.exp(-np.dot(coefficients, xs.T)))

def feed_forward(layer1, weights):
	# first hidden layer
	layer2 = sigmoid(weights[0], layer1)
	print("layer2.shape", layer2.shape)
	print(layer2)

	# output
	layer3 = sigmoid(weights[1], layer2)
	print("layer3", layer3)

	return layer3

def loss_function():
	print("implement. parameters: y, output_layer")

if __name__ == "__main__":
	# input
	layer1 = np.array([1, 2, 3])
	
	weights1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	weights2 = np.array([10, 11, 12])
	# cada weight dentro vai ter um shape!
	weights = [weights1, weights2]

	layer3 = feed_forward(layer1, weights)

	loss_function()

	# derivative_error()

	# backpropagate()

	# update_weights()
