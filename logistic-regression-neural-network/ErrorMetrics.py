import pandas as pd

def count(predicted, y):
	true_positive = 0
	false_positive = 0
	true_negative = 0
	false_negative = 0

	toggle_predicted = []
	for p in predicted:
		if p >= 0.5:
			toggle_predicted.append(1)
		else:
			toggle_predicted.append(0)

	for i in range(len(predicted)):
		if toggle_predicted[i] == y[i]  and y[i] == 1:
			true_positive += 1

		if toggle_predicted[i] == y[i]  and y[i] == 0:
			true_negative += 1

		if toggle_predicted[i] < y[i]:
			false_negative += 1

		if toggle_predicted[i] > y[i]:
			false_positive += 1

	return true_positive, false_positive, true_negative, false_negative

def confusion_matrix(label, predicted, y):
		true_positive, false_positive, true_negative, false_negative = count(predicted, y)
		return pd.DataFrame([[true_positive, false_negative], [false_positive, true_negative]], index=["Real " + label, "Real not " + label], columns=["Predicted " + label, "Predicted not " + label])

def normalized_accuracy(predicted, y):
		true_positive, false_positive, true_negative, false_negative = count(predicted, y)
		return 0.5*((true_positive / (true_positive + false_negative)) + (true_negative / (true_negative + false_positive)))	