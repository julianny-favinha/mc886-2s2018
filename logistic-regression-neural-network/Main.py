import numpy as np
import mnist_reader

# labels = {0: "T-shirt/top", 
# 			1: "Trouser", 
# 			2: "Pullover", 
# 			3: "Dress", 
# 			4: "Coat", 
# 			5: "Sandal", 
# 			6: "Shirt", 
# 			7: "Sneaker", 
# 			8: "Bag", 
# 			9: "Ankle boot"}
labels = {0: "T-shirt/top"}

from LogisticRegression import LogisticRegression

if __name__ == "__main__":
	x_original_train, y_original_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
	x_original_validation, y_original_validation = x_original_train[-10000:], y_original_train[-10000:]
	x_original_train, y_original_train = x_original_train[:-10000], y_original_train[:-10000]

	x_original_test, y_original_test = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='t10k')

	"""
		Suponha que queremos predizer se eh uma label l ou nao. Entao devemos
		1) Trocar todas as labels == l por 1
		2) Trocar todas as labels != l por 0
	"""
	for label in labels:
		y_train = y_original_train.copy()
		y_train[y_original_train == label] = 1
		y_train[y_original_train != label] = 0

		lr = LogisticRegression()
		lr.fit(x_original_train, y_train)