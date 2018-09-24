import numpy as np

def derivative(h, ys, xs, learningRate):
	totalSum = np.dot(h - ys, xs)

	isNanOrIsInf = np.any(np.logical_or(np.isnan(totalSum), np.isinf(totalSum)))
	if isNanOrIsInf:
		raise ValueError("Exception: sum is infinite or not a number.")

	return learningRate * (1 / xs.shape[0]) * totalSum

def descent(initialGuess, model, x, y, learningRate, iterations):
	thetas = initialGuess
	
	for _ in range(iterations):
		h = model(thetas, x)
		thetas = thetas - derivative(h, y, x, learningRate)

	return thetas