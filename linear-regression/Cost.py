import numpy as np

"""
	Calculates the hypotesis based on coefficients
"""
def hypothesis(thetas, xs):
	return np.sum(thetas * xs)

"""
	Function to calculate mean squared error
"""
def cost(thetas, xs, ys):
	m = len(xs)
	sum = 0

	for i in range(m):
		sum += (hypothesis(thetas, xs[i]) - ys[i]) ** 2
	
	return (1/(2*m)) * sum

"""
	Literal cost to view equation
"""
def verboseCost(thetas, xs, ys):
	m = len(xs)
	print("cost = (1/(2*{}))(".format(m), end="")
	for i in range(m):
		if i < m-1:
			print("({} - {})^2 + ".format(hypothesis(thetas, xs[i]), ys[i]), end="")
		else:
			print("({} - {})^2)".format(hypothesis(thetas, xs[i]), ys[i]))