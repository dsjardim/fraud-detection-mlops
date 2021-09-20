# See this Kaggle: https://www.kaggle.com/shivamb/semi-supervised-classification-using-autoencoders
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json
from process_data import process_data


def main():
    data = pd.read_csv("data/creditcard.csv")

    X, y = process_data(data)

    clf = LogisticRegression(solver='liblinear')

    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.25)
    clf.fit(train_x, train_y)
    pred_y = clf.predict(val_x)
    clf_rep = classification_report(val_y, pred_y, output_dict=True)

    metrics_out = {
        "accuracy": clf_rep["accuracy"],
        "precision": clf_rep["weighted avg"]["precision"],
        "recall": clf_rep["weighted avg"]["recall"],
        "f1-score": clf_rep["weighted avg"]["f1-score"]
    }

    with open("data/metrics.json", 'w') as outfile:
        json.dump(metrics_out, outfile)


if __name__ == '__main__':
    main()
