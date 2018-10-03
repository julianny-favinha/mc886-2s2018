import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mnist_reader
import time

from LogisticRegression import LogisticRegression

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

def lr_cost(thetas, xs, ys):	
	sum = np.sum(ys*np.log(LogisticRegression.sigmoid(thetas, xs) + 0.00000001) + (1 - ys)*np.log(1 - LogisticRegression.sigmoid(thetas, xs) + 0.00000001))
	
	return -(1/len(xs)) * sum

"""
	Suppose we want to predict if it is l label or not. Then
	1) Change all labels == l to 1
	2) Change all labels != l to 0
"""
def toggle_class(x, label):
	y = x.copy()
	y[x == label] = 1
	y[x != label] = 0

	return y

if __name__ == "__main__":
	# get training data
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
	initialGuess = np.ones(x_original_train.shape[1])
	learningRate = 0.01
	iterations = 1000
	print("Using learning rate = {} and {} iterations".format(learningRate, iterations))

	predictions = np.array([])
	cost_iterations = []
	for label in labels:
		start_time = time.time()

		lr = LogisticRegression()

		# train
		y_train = toggle_class(y_original_train, label)
		cost_iterations = lr.fit(x_original_train, y_train, labels[label], initialGuess, learningRate, iterations, lr_cost)

		# plot cost by number of iteration graph
		plt.plot([x for x in range(iterations)], cost_iterations, color="blue")
		plt.title("Logistic Regression for label {} with learningRate = {}".format(labels[label], learningRate))
		plt.savefig("CostGraph" + labels[label].replace("/", "-"))
		plt.xlabel("Number of iterations")
		plt.ylabel("Cost")
		plt.clf()

		y_validation = toggle_class(y_original_validation, label)

		# predict
		predicted = lr.predict(x_original_validation, labels[label])
		predictions = np.append(predictions, predicted)

		toggle_predicted = []
		for p in predicted:
			if p >= 0.5:
				toggle_predicted.append(1)
			else:
				toggle_predicted.append(0)

		cm_labels = ["Not " + labels[label], labels[label]]
		cm = confusion_matrix(y_validation.tolist(), toggle_predicted)
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
		plt.savefig("ConfusionMatrix" + labels[label].replace("/", "-"), bbox_inches="tight")
		plt.clf()

		print("Accuracy score = {0:.1f}%".format((accuracy_score(y_validation.tolist(), toggle_predicted) * 100)))

		elapsed_time = time.time() - start_time
		print("Elapsed time: %1f s" %(elapsed_time))

	predictions = predictions.reshape(len(labels.keys()), x_original_validation.shape[0])

	predicted_class = []
	for i in range(predictions.shape[1]):
		column = predictions[:,i]
		predicted_class.append(np.argmax(column))
	
	correct = 0
	for i in range(len(predicted_class)):
		if predicted_class[i] == y_original_validation[i]:
			correct += 1
	print("Predicted correct:", correct)
