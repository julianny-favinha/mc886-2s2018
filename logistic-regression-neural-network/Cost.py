import matplotlib.pyplot as plt

def plot_cost(cost_iterations, label, iterations):
    plt.plot([x for x in range(iterations)], cost_iterations, color="blue")
    plt.title(f'Logistic Regression for label {label} with learningRate = {iterations}')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.savefig('CostGraph' + label.replace("/", "-"))
    plt.clf()