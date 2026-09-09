import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def calculate_accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=4)


def predictions_to_numpy(predictions):
    return np.asarray(predictions)
