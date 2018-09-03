import pandas as pd

"""
	45849 -> training (35849 training and 10000 validation)
	8091 -> test

	carat, depth, table, x, y, z
	price 
"""
def numberVariables():
	file = pd.read_csv("diamonds.csv")

	xs = file.drop(file.columns[file.columns.str.contains("unnamed", case = False)], axis=1)
	xs = xs.drop(["cut", "color", "clarity", "price"], axis=1)

	# normalizing data with min-max normalization
	xs_aux_train = xs[:-8091]
	minimum = xs_aux_train.min()
	maximum = xs_aux_train.max()

	xs_validation = (xs_aux_train[-10000:] - minimum) / (maximum - minimum)
	xs_train = (xs_aux_train[:-10000] - minimum) / (maximum - minimum)
	xs_test = (xs[-8091:] - minimum) / (maximum - minimum)

	y = file["price"]

	y_aux_train = y[:-8091]
	y_validation = y_aux_train[-10000:]
	y_train = y_aux_train[:-10000]
	y_test = y[-8091:]

	return xs_train, xs_validation, xs_test, y_train, y_validation, y_test