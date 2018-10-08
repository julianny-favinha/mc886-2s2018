import matplotlib.pyplot as plt

def plot_cost(cost_iterations, label, iterations, learningRate):
    plt.plot([x for x in range(iterations)], cost_iterations, color="blue")
    plt.title(f'Logistic Regression for label {label} with learningRate = {learningRate}')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.savefig('CostGraph' + label.replace("/", "-"), bbox_inches="tight")
    plt.clf()