import numpy as np
import mnist_reader

from LogisticRegression import LogisticRegression

if __name__ == "__main__":
	x_train, y_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
	x_test, y_test = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='t10k')

	lr = LogisticRegression()
	lr.fit(x_train, y_train)
