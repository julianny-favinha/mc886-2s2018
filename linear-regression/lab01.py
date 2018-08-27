# -*- coding: utf-8 -*-
import numpy as np
import math
class BatchGradientDescent:
	"""
		Attributes
	"""
	coeficients = None
	intercept = None
	
	"""
		Constructor, initializes number of iterations and learning rate 
		Will default do 1000 iterations and learning Rate 1
	"""
	def __init__(self, numIt=100, lr=0.5):
		self.maxIteration = numIt
		self.learningRate = lr

	"""
		Clculates the hypotesis based on coeficients
	"""
	def hypothesis(self, thetas, xs):
		return np.sum([t*x for t, x in zip(thetas, xs)])

	"""
		
	"""
	def cost(self, thetas, xs, ys):
		m = len(xs)
		sum = 0
		
		for i in range(m):
			sum += (self.hypothesis(thetas, xs[i]) - ys[i]) ** 2
		
		return (1/(2*m)) * sum

	"""
		
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
		
	"""
	def normalEq(self):
		print("normalEq not implemented")
		pass
	
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
	xs = np.array([[1, 3], [1, 3.5], [1, 2]])
	ys = np.array([6, 8, 5])
	
	#h = [hypothesis(thetas, x) for x in xs]
	#print("htheta(xs) = {}".format(h))
	#verbose_cost(thetas, xs, ys)
	#print("cost = {}".format(cost(thetas, xs, ys)))
	aux = BatchGradientDescent(1000,0.1)
	aux.fit(xs, ys,thetas)
	print(aux.coeficients)