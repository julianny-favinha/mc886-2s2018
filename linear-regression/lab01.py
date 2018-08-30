# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

class BatchGradientDescent:
	"""
		Attributes
	"""
	coeficients = None
	intercept = None
	
	"""
		Constructor, initializes number of iterations and learning rate 
		Will default do 100 iterations and learning Rate 0.5
	"""
	def __init__(self, numIt=100, lr=0.5):
		self.maxIteration = numIt
		self.learningRate = lr

	"""
		Calculates the hypotesis based on coeficients
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
		self.coeficients = thetas

	"""
		Normal Equation method to find coeffiecients of hypothesis
	"""
	def normalEq(self, xs, ys):
		 # theta = inverse(Xt * X) * Xt * y
		xst = xs.transpose()
		inverse = np.linalg.inv(xst.dot(xs))
		return (inverse.dot(xst)).dot(ys)
	
	#----------------------------------------------------------------------------------------------
	
if __name__ == "__main__":
	""" 
	usando variaveis x0 e x1 (sendo que x0 é sempre 1)
	suponha que encontrei a hipotese htheta(x) = 0.5*x0 + 2*x1
	ou seja, parametros sao theta0 = 0.5 e theta1 = 2
	
	tabela de dados
	x0		x1		y
	1		 3		 6
	1		 3.5	 8
	1		 2		 5
	"""
	thetas = np.array([0.5, 2])
	xs = np.array([[1, 3], [1, 3.5], [1, 2], [1, 7]])
	ys = np.array([6, 8, 5, 9])

	# cost by number of iterations
	iterations = np.array([1, 10, 100, 1000, 10000, 100000])
	cost = []

	for it in iterations:
		bgd = BatchGradientDescent(it, 0.1)
		bgd.fit(xs, ys, thetas)
		batch_coef = bgd.coeficients
		cost.append(bgd.cost(batch_coef, xs, ys))

	plt.plot(iterations, cost, color="blue")
	plt.xticks(iterations)
	plt.xlabel("Number of iterations")
	plt.yticks()
	plt.ylabel("Cost")
	plt.show()

	# normal equation
	bgd_normalEq = BatchGradientDescent()
	normalEq_coef = bgd_normalEq.normalEq()