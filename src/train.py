# See this Kaggle: https://www.kaggle.com/shivamb/semi-supervised-classification-using-autoencoders
# See this Kaggle: https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
import json
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from src.preprocessing import prepare_dataset


def main():
    data = pd.read_csv("data/creditcard.csv")

    data_dict = prepare_dataset(data, False)

    X_train = data_dict["X_train"]
    X_valid = data_dict["X_valid"]
    y_train = data_dict["y_train"]
    y_valid = data_dict["y_valid"]

    clf = LogisticRegression()

    clf.fit(X_train, y_train)
    training_score = cross_val_score(clf, X_train, y_train, cv=5)
    print(clf.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

    # Logistic Regression
    log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
    grid_log_reg.fit(X_train, y_train)

    log_reg_clf = grid_log_reg.best_estimator_

    pred_y = log_reg_clf.predict(X_valid)
    clf_rep = classification_report(y_valid, pred_y, output_dict=True)

    cm = confusion_matrix(y_valid, pred_y, normalize='true')
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
