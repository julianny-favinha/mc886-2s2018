import numpy as np

class BatchGradientDescent:
	"""
		Attributes
	"""
	coefficients = None
	intercept = None
	
	"""
		Constructor, initializes number of iterations and learning rate 
		Will default do 100 iterations and learning Rate 0.5
	"""
	def __init__(self, numIt=100, lr=0.5):
		self.maxIteration = numIt
		self.learningRate = lr

	def descent(self, h, ys, xs):
		totalSum = np.dot(h - ys, xs)

		isNanOrIsInf = np.any(np.logical_or(np.isnan(totalSum), np.isinf(totalSum)))
		if isNanOrIsInf:
			raise ValueError("Exception: sum is infinite or not a number.")

		return self.learningRate * (1/len(xs)) * totalSum

	"""
		Find coefficients of hypothesis using Gradient Descent
	"""
	def fit(self, xs, ys, initialGuess):
		thetas = initialGuess
		for _ in range(self.maxIteration):
			h = np.sum(xs * thetas, axis=1)
			thetas = thetas - self.descent(h, ys, xs)
	
		self.coefficients = thetas