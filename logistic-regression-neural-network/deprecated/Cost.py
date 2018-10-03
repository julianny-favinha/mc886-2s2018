import numpy as np

from LogisticRegression import LogisticRegression
from MultinomialLogisticRegression import MultinomialLogisticRegression

def lr_cost(thetas, xs, ys):	
	sum = np.sum(ys*np.log(LogisticRegression.sigmoid(thetas, xs) + 0.00000001) + (1 - ys)*np.log(1 - LogisticRegression.sigmoid(thetas, xs) + 0.00000001))
	
	return -(1/len(xs)) * sum

def cross_entropy(thetas, xs, ys):
	predicted = MultinomialLogisticRegression.softmax(thetas, xs)
	sum = np.sum(predicted * np.log(ys.T + 0.00000001))
	return -(1/ys.shape[1]) * sum