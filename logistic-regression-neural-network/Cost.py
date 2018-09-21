import numpy as np

def hypothesis(thetas, xs):
	return np.sum(thetas * xs)

def cost(thetas, xs, ys):
	m = len(xs)
	sum = 0

	for i in range(m):
		sum += ys[i]*np.log(hypothesis(thetas, xs[i])) + (1 - ys[i])*np.log(1 - hypothesis(thetas, xs[i]))
	
	return -(1/m) * sum