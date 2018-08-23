import numpy as np
import matplotlib.pyplot as plt

def hypothesis(thetas, xs):
	return np.sum([t*x for t, x in zip(thetas, xs)])

def cost_function(thetas, xs, ys):
	m = len(xs)
	sum = 0
	
	for i in range(m):
		sum += (hypothesis(thetas, xs[i]) - ys[i][0]) ** 2
	
	return (1/(2*m)) * sum

def verbose_cost_function(thetas, xs, ys):
	m = len(xs)
	print("cost function = (1/(2*{}))(".format(m), end="")
	for i in range(m):
		if i < m-1:
			print("({} - {})^2 + ".format(hypothesis(thetas, xs[i]), ys[i][0]), end="")
		else:
			print("({} - {})^2".format(hypothesis(thetas, xs[i]), ys[i][0]), end="")
	print(")")

""" 
TESTE
usando variaveis x0 e x1 (sendo que x0 é sempre 1)
suponha que encontrei a hipotese htheta(x) = 0.5*x0 + 2*x1
ou seja, parametros sao theta0 = 0.5 e theta1 = 2

tabela de dados
x0    x1    y
1     3     6
1     3.5   8
1     2     5
"""
thetas = np.array([0.5, 2])
xs = np.array([[1, 3], [1, 3.5], [1, 2]])
ys = np.array([[6], [8], [5]])

h = [hypothesis(thetas, x) for x in xs]

print("htheta(xs) = {}".format(h))
verbose_cost_function(thetas, xs, ys)
print("cost_function = {}".format(cost_function(thetas, xs, ys)))
