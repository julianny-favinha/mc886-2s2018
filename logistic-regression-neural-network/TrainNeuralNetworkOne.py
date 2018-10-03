import numpy as np
import matplotlib.pyplot as plt
import mnist_reader
import time

from NeuralNetworkOne import NeuralNetwork

from sklearn.metrics import confusion_matrix, accuracy_score

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

	start_time = time.time()

	NN = NeuralNetwork()
	learningRate = 0.01

	print("weights1 before")
	print(NN.weights1)
	print("weights2 before")
	print(NN.weights2)
	
	for i in range(100):
		a1_acum_delta2 = 0
		a2_acum_delta3 = 0
		loss = 0

		mini_batch = 500
		print("Performing feed forward for images {} to {}...".format(i*mini_batch, (i+1)*mini_batch - 1))

		for batch in range(0, x_layer_train.shape[0], mini_batch):
			a3 = NN.forward(x_layer_train[batch:batch+mini_batch])
			# print("a3.shape", a3.shape)
			# print("a3")
			# print(a3[0])
			# print(np.sum(a3, axis=1).shape)

			# calculate cross_entropy function
			# loss += NN.CrossEntropy(a3, y_hot_encoding[batch:batch+mini_batch])

			# update deltas with the derivative cost function
			a1delta2, a2delta3 = NN.CrossEntropyDerivative(x_layer_train[batch:batch+mini_batch], y_hot_encoding[batch:batch+mini_batch])
			a1_acum_delta2 += a1delta2
			a2_acum_delta3 += a2delta3
			#NN.bias2 -= learningRate * deltabias2

		# calculate cross entropy loss
		a3_train = NN.forward(x_layer_train)
		loss = NN.CrossEntropy(a3_train, y_hot_encoding)
		print("Cross entropy", loss)

		# update weights
		NN.weights1 -= learningRate * a1_acum_delta2 / (x_layer_train.shape[0] / mini_batch)
		NN.weights2 -= learningRate * a2_acum_delta3 / (x_layer_train.shape[0] / mini_batch)
		
		print("weights1")
		print(NN.weights1)

		# print("weights2")
		# print(NN.weights2)

	print("Predicting values for validation set...")
	# print(x_layer_validation.shape)
	predictions = NN.forward(x_layer_validation)
	# print(predictions.shape)
	# print(predictions[0])

	predicted_class = []
	for pred in predictions:
		index = np.argmax(pred)
		predicted_class.append(index)

	print(predicted_class[0:10])
	print(y_layer_validation[0:10])

	cm_labels = list(labels.keys())
	cm = confusion_matrix(y_layer_validation.tolist(), predicted_class)
	print("Confusion matrix")
	print(cm)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(cm)
	plt.title('Confusion matrix')
	fig.colorbar(cax)
	ax.set_xticklabels([''] + cm_labels)
	ax.set_yticklabels([''] + cm_labels)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.savefig("ConfusionMatrixNeuralNetwork", bbox_inches="tight")
	plt.clf()

	print("Accuracy score = {0:.1f}%".format((accuracy_score(y_layer_validation.tolist(), predicted_class) * 100)))

	elapsed_time = time.time() - start_time
	print("Elapsed time: %1f s" %(elapsed_time))