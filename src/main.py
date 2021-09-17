# See this Kaggle: https://www.kaggle.com/shivamb/semi-supervised-classification-using-autoencoders
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from model import AutoEncoder
from process_data import process_data

np.random.seed(203)


def main():
    data = pd.read_csv("data/creditcard.csv")

    x_raw, y_raw, x_norm, x_fraud = process_data(data)

    model = AutoEncoder(input_shape=x_raw.shape[1],
                        optimizer="adadelta",
                        loss="mse",
                        batch_size=256,
                        epochs=10,
                        shuffle=True,
                        validation_split=0.20)

    model.fit(x_norm[0:2000], x_norm[0:2000])
    hidden_repr = model.get_representation()

    norm_hid_rep = hidden_repr.predict(x_norm[:3000])
    fraud_hid_rep = hidden_repr.predict(x_fraud)

    rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis=0)
    y_n = np.zeros(norm_hid_rep.shape[0])
    y_f = np.ones(fraud_hid_rep.shape[0])
    rep_y = np.append(y_n, y_f)

    train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, test_size=0.25)
    clf = LogisticRegression(solver="lbfgs").fit(train_x, train_y)
    pred_y = clf.predict(val_x)

    print("")
    print("Classification Report: ")
    print(classification_report(val_y, pred_y))

    print("")
    print("Accuracy Score: ", accuracy_score(val_y, pred_y))


if __name__ == '__main__':
    main()
