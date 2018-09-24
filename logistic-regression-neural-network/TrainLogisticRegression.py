import numpy as np
import mnist_reader
import time

from sklearn.preprocessing import MinMaxScaler
from skimage import io, color, measure, exposure
import matplotlib.pyplot as plt

from LogisticRegression import LogisticRegression

from Cost import cost

# labels = {0: "T-shirt/top", 
#  			1: "Trouser", 
#  			2: "Pullover", 
#  			3: "Dress", 
#  			4: "Coat", 
#  			5: "Sandal", 
#  			6: "Shirt", 
#  			7: "Sneaker", 
#  			8: "Bag",
#  			9: "Ankle boot"}
labels = {0: "T-shirt/top", 
 			1: "Trouser",
 			2: "Pullover",
 			3: "Dress"}

if __name__ == "__main__":
	x_original_train, y_original_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
	x_original_validation, y_original_validation = x_original_train[-10000:], y_original_train[-10000:]
	x_original_train, y_original_train = x_original_train[:-10000], y_original_train[:-10000]

	scaler = MinMaxScaler()
	scaler.fit(x_original_train.astype(float))
	x_original_train = scaler.transform(x_original_train.astype(float))
	x_original_validation = scaler.transform(x_original_validation.astype(float))


	x_original_train = np.c_[ np.ones(x_original_train.shape[0]) , x_original_train ]
	x_original_validation = np.c_[ np.ones(x_original_validation.shape[0]) , x_original_validation ]

	lr = LogisticRegression()

	predictions = np.array([])
	for label in labels:
		start_time = time.time()

		"""
			Suponha que queremos predizer se eh uma label l ou nao. Entao devemos
			1) Trocar todas as labels == l por 1
			2) Trocar todas as labels != l por 0
		"""
		y_train = y_original_train.copy()
		y_train[y_original_train == label] = 1
		y_train[y_original_train != label] = 0

		lr.fit(x_original_train, y_train)

		print(lr.coefficients)

		y_validation = y_original_validation.copy()
		y_validation[y_original_validation == label] = 1
		y_validation[y_original_validation != label] = 0

		predictions = np.append(predictions, lr.predict(x_original_validation))

		elapsed_time = time.time() - start_time
		print("Elapsed time: %1f s" %(elapsed_time))

	# print(labels.keys())
	# print(cost_list)
	# plt.plot(labels.keys(), cost_list, color="blue")
	# plt.savefig("CostGraph")

	predictions = predictions.reshape(len(labels.keys()), x_original_validation.shape[0])
	# print("predictions", predictions)
	# print("predictions shape", predictions.shape)

	predicted_class = []
	for i in range(predictions.shape[1]):
		column = predictions[:,i]
		predicted_class.append(np.argmax(column))
	# print("predicted_class", predicted_class)
	
	correct = 0
	for i in range(len(predicted_class)):
		if predicted_class[i] == y_original_validation[i]:
			correct += 1
	print("correct", correct)
