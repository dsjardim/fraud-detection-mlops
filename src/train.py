# See this Kaggle: https://www.kaggle.com/shivamb/semi-supervised-classification-using-autoencoders
import json
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from process_data import process_data


def main():
    data = pd.read_csv("data/creditcard.csv")

    X, y = process_data(data)

    clf = LogisticRegression()

    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.25)
    clf.fit(train_x, train_y)
    pred_y = clf.predict(val_x)
    clf_rep = classification_report(val_y, pred_y, output_dict=True)

    cm = confusion_matrix(val_y, pred_y, normalize='true')
    hm = sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    fig = hm.get_figure()
    fig.savefig('data/confusion_matrix.png', dpi=400)

    metrics_out = {
        "accuracy": clf_rep["accuracy"],
        "precision": clf_rep["weighted avg"]["precision"],
        "recall": clf_rep["weighted avg"]["recall"],
        "f1-score": clf_rep["weighted avg"]["f1-score"]
    }

    with open("data/metrics.json", 'w') as outfile:
        json.dump(metrics_out, outfile)

    with open('data/model.pickle', 'wb') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':
    main()
