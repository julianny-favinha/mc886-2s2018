import numpy as np

def derivative(h, ys, xs, learningRate):
	totalSum = np.dot(h - ys, xs)

	isNanOrIsInf = np.any(np.logical_or(np.isnan(totalSum), np.isinf(totalSum)))
	if isNanOrIsInf:
		raise ValueError("Exception: sum is infinite or not a number.")

	return learningRate * (1/len(xs)) * totalSum

def descent(initialGuess, model, x, y, learningRate):
	thetas = initialGuess
	
	for _ in range(1000): # MELHORAR: quantas iteracoes?
		h = np.sum(x * thetas, axis=1)
		thetas = thetas - derivative(h, y, x, learningRate)

	return thetas