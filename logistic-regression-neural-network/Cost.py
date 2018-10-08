import matplotlib.pyplot as plt

def plot_cost(cost_iterations, title, file_name, iterations, learningRate):
    plt.plot([x for x in range(iterations)], cost_iterations, color="blue")
    plt.title(f'Logistic Regression for {title}, learningRate = {learningRate}')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.savefig('CostGraph' + file_name.replace("/", "-"), bbox_inches="tight")
    plt.clf()