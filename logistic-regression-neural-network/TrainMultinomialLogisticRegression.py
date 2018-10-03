import numpy as np
import matplotlib.pyplot as plt
import mnist_reader
import time

from MultinomialLogisticRegression import MultinomialLogisticRegression

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

def cross_entropy(thetas, xs, ys):
	predicted = MultinomialLogisticRegression.softmax(thetas, xs)
	sum = np.sum(predicted * np.log(ys.T + 0.00000001))
	return -(1/ys.shape[1]) * sum

if __name__ == "__main__":
	# get the training data
	x_original_train, y_original_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
	x_original_validation, y_original_validation = x_original_train[-10000:], y_original_train[-10000:]
	x_original_train, y_original_train = x_original_train[:-10000], y_original_train[:-10000]

	# pixels normalized to [0, 1] interval
	x_original_train = x_original_train / 255
	x_original_validation = x_original_validation / 255

	# add 1 column to x
	x_original_train = np.c_[np.ones(x_original_train.shape[0]), x_original_train]
	x_original_validation = np.c_[np.ones(x_original_validation.shape[0]), x_original_validation]

	# initial values
	initialGuess = np.ones((len(labels), x_original_train.shape[1]))
	learningRate = 0.0001
	iterations = 1000
	print("Using learning rate = {} and {} iterations".format(learningRate, iterations))

	start_time = time.time()

	print("Creating one hot encoding for target...")
	y_hot_encoding = list()
	for index in y_original_train:
		y_line =  [0 for _ in range(len(labels))]
		y_line[index] = 1
		y_hot_encoding.append(y_line)
	y_hot_encoding = np.array(y_hot_encoding)
	
	# train
	mlr = MultinomialLogisticRegression()
	cost_iterations = mlr.fit(x_original_train, y_hot_encoding.T, "Multinomial", initialGuess, learningRate, iterations, cross_entropy)

	plt.plot([x for x in range(iterations)], cost_iterations, color="blue")
	plt.title("Multinomial Logistic Regression with learningRate = {}".format(learningRate))
	plt.savefig("CostGraphMultinomial")
	plt.xlabel("Number of iterations")
	plt.ylabel("Cost")
	plt.clf()

	# predict
	predictions = (mlr.predict(x_original_validation, "Multinomial"))
	# print(predictions[0])

	# find predicted class
	predicted_class = []
	for pred in predictions:
		ind = np.argmax(pred)
		predicted_class.append(ind)

	cm_labels = list(labels.keys())
	cm = confusion_matrix(y_original_validation.tolist(), predicted_class)
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
	plt.savefig("ConfusionMatrixMultinomial", bbox_inches="tight")
	plt.clf()

	print("Accuracy score = {0:.1f}%".format((accuracy_score(y_original_validation.tolist(), predicted_class) * 100)))

	elapsed_time = time.time() - start_time
	print("Elapsed time: %1f s" %(elapsed_time))