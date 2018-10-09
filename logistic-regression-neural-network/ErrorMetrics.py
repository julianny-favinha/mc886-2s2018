import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score

def show_metrics(predicted, title, labels, y, file_name):
    cm = confusion_matrix(y, predicted)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title("Confusion matrix " + title)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(file_name.replace("/", "-"), bbox_inches="tight")
    plt.clf()

    print("Accuracy score = {0:.1f}%".format((accuracy_score(y, predicted) * 100)))