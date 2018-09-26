import numpy as np

def hyp(coefficients, xs):
	return 1 / (1 + np.exp(-np.dot(coefficients, xs.T)))

def cost(thetas, xs, ys):	
	sum = np.sum(ys*np.log(hyp(thetas, xs) + 0.00000001) + (1 - ys)*np.log(1 - hyp(thetas, xs) + 0.00000001))
	
	return -(1/len(xs)) * sum