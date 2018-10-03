import numpy as np

def derivative(h, ys, xs, learningRate):
	totalSum = np.dot(h.T - ys, xs)

	isNanOrIsInf = np.any(np.logical_or(np.isnan(totalSum), np.isinf(totalSum)))
	if isNanOrIsInf:
		raise ValueError("Exception: sum is infinite or not a number.")

	return learningRate * (1 / xs.shape[0]) * totalSum

def descent(initialGuess, model, x, y, learningRate, iterations, costFunction):
	thetas = initialGuess

	cost_iterations = []
	for it in range(iterations):
		h = model(thetas, x)
		thetas = thetas - derivative(h, y, x, learningRate)

		cost_iterations.append(costFunction(thetas, x, y))

	return thetas, cost_iterations