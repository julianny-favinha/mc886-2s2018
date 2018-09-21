import numpy as np
import mnist_reader

if __name__ == "__main__":
	x_train, y_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
	x_test, y_test = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='t10k')