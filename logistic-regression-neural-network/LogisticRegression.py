import numpy as np
from GradientDescent import descent

class LogisticRegression:
	coefficients = None # MELHORAR

	"""
		Find coefficients
	"""
	def fit(self, x, y, options={}):
		print("Performing fit...")

		initialGuess = np.ones(x.shape[1]) # MELHORAR: quais valores de theta comecar?
		learningRate = 0.0000001 # MELHORAR: qual valor?

		self.coefficients = descent(initialGuess, self.coefficients, x, y, learningRate)
		print(self.coefficients)

	"""
		Uses model (coefficients) to predict y given x
	"""
	def predict(self, xs, ys):
		correct = 0

		print("Performing predictions...")
		for i in range(xs.shape[0]):
			prediction = 1 / (1 + np.exp(-np.sum(self.coefficients * xs[i])))
			if prediction >= 0.5:
				prediction = 1
			else:
				prediction = 0

			if prediction == ys[i]:
				correct += 1

		print("Number of correct predictions:", correct)

