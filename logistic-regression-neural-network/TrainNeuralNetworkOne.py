import numpy as np
import mnist_reader

from NeuralNetworkOne import NeuralNetwork
# from matplotlib.pyplot import plot, grid, legend, show

labels = {0: "T-shirt/top", 
 			1: "Trouser", 
 			2: "Pullover", 
 			3: "Dress", 
 			4: "Coat", 
 			5: "Sandal", 
 			6: "Shirt", 
 			7: "Sneaker", 
 			8: "Bag",
 			9: "Ankle boot"}

def softmax(z):
	exponent = np.exp(z)
	return exponent / (np.sum(exponent))

if __name__ == "__main__":
	x_layer_train, y_layer_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
	x_layer_validation, y_layer_validation = x_layer_train[-10000:], y_layer_train[-10000:]
	x_layer_train, y_layer_train = x_layer_train[:-10000], y_layer_train[:-10000]

	x_layer_train = x_layer_train / 255
	x_layer_validation = x_layer_validation / 255

	print("Creating one hot encoding for target...")
	y_hot_encoding = list()
	for index in y_layer_train:
		y_line =  [0 for _ in range(len(labels))]
		y_line[index] = 1
		y_hot_encoding.append(y_line)
	y_hot_encoding = np.array(y_hot_encoding)

	NN = NeuralNetwork()

	for _ in range(100):
		output_layer = NN.forward(x_layer_train)
		print("output_layer", output_layer)

		output_layer_softmax = softmax(NN.z3)
		print("softmax", output_layer_softmax)

		loss = NN.lossFunction(output_layer_softmax, y_hot_encoding)
		print("loss", loss)

		dJdW1, dJdW2 = NN.JPrime(x_layer_train, y_hot_encoding) 

		learningRate = 1

		NN.weights1 -= learningRate * dJdW1
		NN.weights2 -= learningRate * dJdW2
		print("dJdW1")
		print(dJdW1)
		print("weights1")
		print(NN.weights1)

	# test = np.arange(-5, 5, 0.1)
	# plot(test, NN.sigmoid(test))
	# plot(test, sigmoidDerivative(test))
	# grid(1)
	# legend(['sigmoid', 'derivative'])
	# show()