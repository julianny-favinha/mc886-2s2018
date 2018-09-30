import numpy as np
from matplotlib.pyplot import plot, grid, legend, show

input_layer = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
y = np.array([[1, 0], [0, 1], [0, 1]])

NN = NeuralNetwork()

output_layer = NN.forward(input_layer)
print("output_layer", output_layer)

loss = NN.lossFunction(output_layer, y)
print("loss", loss)

dJdW1, dJdW2 = NN.JPrime(input_layer, y) 
print(dJdW1)
print(dJdW2)

learningRate = 1

NN.weights1 -= learningRate * dJdW1
NN.weights2 -= learningRate * dJdW2

# test = np.arange(-5, 5, 0.1)
# plot(test, NN.sigmoid(test))
# plot(test, sigmoidDerivative(test))
# grid(1)
# legend(['sigmoid', 'derivative'])
# show()