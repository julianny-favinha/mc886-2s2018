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

	"""
		Calculates the hypotesis based on coefficients
	"""
	def hypothesis(self, thetas, xs):
		return np.sum(thetas * xs)

	"""
		Function to calculate mean squared error
	"""
	def cost(self, thetas, xs, ys):
		m = len(xs)
		sum = 0
		
		for i in range(m):
			sum += (self.hypothesis(thetas, xs[i]) - ys[i]) ** 2
		
		return (1/(2*m)) * sum

	"""
		Literal cost to view equation
	"""
	def verboseCost(self, thetas, xs, ys):
		m = len(xs)
		print("cost = (1/(2*{}))(".format(m), end="")
		for i in range(m):
			if i < m-1:
				print("({} - {})^2 + ".format(self.hypothesis(thetas, xs[i]), ys[i]), end="")
			else:
				print("({} - {})^2)".format(self.hypothesis(thetas, xs[i]), ys[i]))

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

	"""
		Normal Equation method to find coeffiecients of hypothesis
		theta = inverse(Xt * X) * Xt * y
	"""
	def normalEq(self, xs, ys):
		xst = xs.transpose()
		inverse = np.linalg.inv(xst.dot(xs))
		return (inverse.dot(xst)).dot(ys)