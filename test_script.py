from sklearn import svm
import numpy as np
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def getF1Score(y_test, y_pred):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i, (test, pred) in enumerate(zip(y_test, y_pred)):
        if test == pred and pred == 1:
            true_positives += 1
        elif test == pred and pred == -1:
            true_negatives += 1
        elif test != pred and pred == 1:
            false_positives += 1
        elif test != pred and pred == -1:
            false_negatives += 1
        else:
            raise AssertionError("Unexpected value:", test, pred)
    return 2 * true_positives / (2 * true_positives + false_positives + false_negatives)

