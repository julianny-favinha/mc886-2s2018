import numpy as np

class NeuralNetwork():
    weights = None

    def __init__(self, learningRate, activationFunction, activationDerivativeFunction):
        self.learning_rate = learningRate

        self.activation_function = activationFunction
        self.activation_derivative_function = activationDerivativeFunction


    def init_network(self, shape):
        weights_array = []
        for i in range(0, len(shape) - 1):
            layer_weight = np.random.rand(shape[i + 1], shape[i])
            weights_array.append(layer_weight)

        self.weights = weights_array
        return weights_array


    def predict(self, x, weights):
        return self.softmax(self.forward(x, weights)[-1])


    def forward(self, x, weights):
        current_input = x
        outputs = []

        for network_weight in weights:
            z = np.dot(current_input, network_weight.T)
            current_output = self.activation_function(z)
            outputs.append(current_output)
            current_input = current_output

        return outputs


    def update_weights(self, x, y, deltas, weights):
        new_weights = weights

        for layer in range(len(deltas)):
            if layer - 1 < 0:
                input_used = x
            else:
                input_used = y[layer - 1]

            z = np.dot(input_used.T, deltas[layer])
            new_weights[layer] += self.learning_rate * z.T

        return new_weights


    def calculate_deltas(self, target, predictions, weights):
        deltas = []

        final_layer_error = target - predictions[-1]
        final_layer_delta = final_layer_error * self.activation_derivative_function(predictions[-1])

        deltas.append(final_layer_delta)
        back_idx = len(predictions) - 2

        current_delta = final_layer_delta
        for network_weight in weights[::-1][:-1]:
            next_error = np.dot(current_delta, network_weight)
            next_delta = next_error * self.activation_derivative_function(predictions[back_idx])
            deltas.append(next_delta)

            current_delta = next_delta
            back_idx -= 1

        return deltas[::-1]


    def back_propagate(self, x, y, predictions, weights):
        deltas = self.calculate_deltas(y, predictions, weights)

        return self.update_weights(x, predictions, deltas, weights)


    def iterate(self, x, y, weights): 
        current_input = x
        current_weights = weights

        predictions = self.forward(x, current_weights)
        error = self.cost(predictions[-1], y)
        print("Error =", error)
        new_weights = self.back_propagate(x, y, predictions, current_weights)
        
        return error, new_weights


    def train(self, x, y, epochs, initial_weights):
        weights = initial_weights
        cost_iterations = []

        for epoch in range(epochs):
            print("Forwarding for epoch {}/{}...".format(epoch + 1, epochs))
            error, weights = self.iterate(x, y, weights)
            cost_iterations.append(error)
        
        return cost_iterations, weights


    def softmax(self, z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        return e_x / div


    def cost(self, h, y):
        h = h / np.amax(h)
        return (-1.0 / h.shape[0]) * np.sum(y * np.log(h + 0.0001) + (1.0 - y) * np.log(1.0 - h + 0.0001))
