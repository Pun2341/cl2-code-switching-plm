from sklearn import svm
import numpy as np
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import random

# Temp data
models = ["bert-base-multilingual-cased", "xlm-roberta-base", "xlm-roberta-large"]
num_layers = 24
# (y_test, y_pred) per layer?
layers = [[]]*len(models)
for x in range(len(models)):
    acc = []
    for j in range(num_layers):
        acc.append(([random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)],
                    [random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]))
    layers[x] = acc


f1Scores = np.zeros((len(models), num_layers))
for i, model in enumerate(models):
    for j, (y_test, y_pred) in enumerate(layers[i]):
        f1 = f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', zero_division=0.0)
        # Need to figure out how to get the average over "5 seeds"
        f1Scores[i, j] = f1

    plt.plot(range(1, num_layers+1), f1Scores[i, :], label=model)
print(layers)
# Add labels and title
plt.xlabel('Layers')
plt.ylabel('F-1 Score')
plt.title('Mean F-1 Scores across layers for different PLMs')

# Add a legend
plt.legend()

# Show the plot
plt.show()
