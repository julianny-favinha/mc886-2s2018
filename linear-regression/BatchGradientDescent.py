import numpy as np
import math

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
		return np.sum([t*x for t, x in zip(thetas, xs)])

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
	def verbose_cost(self, thetas, xs, ys):
		m = len(xs)
		print("cost = (1/(2*{}))(".format(m), end="")
		for i in range(m):
			if i < m-1:
				print("({} - {})^2 + ".format(self.hypothesis(thetas, xs[i]), ys[i]), end="")
			else:
				print("({} - {})^2)".format(self.hypothesis(thetas, xs[i]), ys[i]))

	"""
		Find coefficients of hypothesis using Gradient Descent
	"""
	def fit(self, xs, ys, initialGuess):
		thetas = initialGuess
		for _ in range(self.maxIteration):
			h = [self.hypothesis(thetas, x) for x in xs]
			n = len(thetas)
			currentThetas = np.empty(n, dtype=np.float64)
			for j in range(n):
				currentThetas[j] = 0
				somatoria = 0
				m = len(xs)
				for i in range(m):
					somatoria += (h[i] - ys[i]) * xs[i][j]
				if math.isnan(somatoria) or math.isinf(somatoria):
					raise ValueError("somatória se tornou infinita!")
				currentThetas[j] += thetas[j] - (self.learningRate * (1/m) * somatoria)
	
			thetas = currentThetas
		self.coefficients = thetas

	"""
		Normal Equation method to find coeffiecients of hypothesis
		theta = inverse(Xt * X) * Xt * y
	"""
	def normalEq(self, xs, ys):
		xst = xs.transpose()
		inverse = np.linalg.inv(xst.dot(xs))
		return (inverse.dot(xst)).dot(ys)