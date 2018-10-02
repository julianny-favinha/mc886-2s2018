import numpy as np
import mnist_reader

from NeuralNetworkOne import NeuralNetwork

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

if __name__ == "__main__":
	x_layer_train, y_layer_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
	x_layer_validation, y_layer_validation = x_layer_train[-10000:], y_layer_train[-10000:]
	x_layer_train, y_layer_train = x_layer_train[:-10000], y_layer_train[:-10000]

	x_layer_train = x_layer_train / 255
	x_layer_train = np.c_[np.ones(x_layer_train.shape[0]), x_layer_train]
	x_layer_validation = x_layer_validation / 255
	x_layer_validation = np.c_[np.ones(x_layer_validation.shape[0]), x_layer_validation]

	print("Creating one hot encoding for target...")
	y_hot_encoding = list()
	for index in y_layer_train:
		y_line = [0 for _ in range(len(labels))]
		y_line[index] = 1
		y_hot_encoding.append(y_line)
	y_hot_encoding = np.array(y_hot_encoding)
	# print("y_hot_encoding.shape", y_hot_encoding.shape)

	NN = NeuralNetwork()
	learningRate = 0.1

	# print("weights1 before")
	# print(NN.weights1)
	# print("weights2 before")
	# print(NN.weights2)

	for i in range(10):
		print(i)
		for batch in range(0, 49990, 10):
			output_layer = NN.forward(x_layer_train[batch:batch+10])
			# print("output_layer.shape", output_layer.shape)
			# print("output_layer")
			# print(output_layer[0])
			# print(np.sum(output_layer, axis=1).shape)

			loss = NN.J(output_layer, y_hot_encoding[batch:batch+10])
			# print("lossFunction", loss)

			deltabias2, a1delta2, a2delta3 = NN.JPrime(x_layer_train[batch:batch+10], y_hot_encoding[batch:batch+10])
			# print(a1delta2)
			NN.weights1 -= 0.1 * learningRate * a1delta2
			NN.weights2 -= 0.1 * learningRate * a2delta3
			NN.bias2 -= 0.1 * learningRate * deltabias2

		preds = NN.softmax(NN.forward(x_layer_validation[-10:]))
		print("preds")
		for pred in preds:
			print(pred)

	print("weights1 after")
	print(a1delta2)
	print("weights2 after")
	print(a2delta3)
	preds = NN.forward(x_layer_validation[-10:])
	print("preds")
	for pred in preds:
		print(pred)

	y_hot_encoding = list()
	for index in y_layer_validation:
		y_line = [0 for _ in range(len(labels))]
		y_line[index] = 1
		y_hot_encoding.append(y_line)
	y_hot_encoding = np.array(y_hot_encoding)
	real_y = y_hot_encoding[-10:]
	print("real_y")
	for pred in real_y:
		print(pred)