# See this Kaggle: https://www.kaggle.com/shivamb/semi-supervised-classification-using-autoencoders
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import torch.nn

np.random.seed(203)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

    def forward(self, x):
        pass


def train():
    data = pd.read_csv("data/creditcard.csv")

    data["Time"] = data["Time"].apply(lambda x: x / 3600 % 24)
    data.head()

    vc = data['Class'].value_counts().to_frame().reset_index()
    vc['percent'] = vc["Class"].apply(lambda x: round(100 * float(x) / len(data), 2))
    vc = vc.rename(columns={"index": "Target", "Class": "Count"})

    non_fraud = data[data['Class'] == 0].sample(1000)
    fraud = data[data['Class'] == 1]

    df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
    X = df.drop(['Class'], axis=1).values
    Y = df["Class"].values

    model = AutoEncoder()

    ## input layer
    input_layer = Input(shape=(X.shape[1],))

    ## encoding part
    encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoded = Dense(50, activation='relu')(encoded)

    ## decoding part
    decoded = Dense(50, activation='tanh')(encoded)
    decoded = Dense(100, activation='tanh')(decoded)

    ## output layer
    output_layer = Dense(X.shape[1], activation='relu')(decoded)

    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer="adadelta", loss="mse")

    x = data.drop(["Class"], axis=1)
    y = data["Class"].values

    x_scale = preprocessing.MinMaxScaler().fit_transform(x.values)
    x_norm, x_fraud = x_scale[y == 0], x_scale[y == 1]

    autoencoder.fit(x_norm[0:2000], x_norm[0:2000],
                    batch_size=256, epochs=10,
                    shuffle=True, validation_split=0.20)

    hidden_representation = Sequential()
    hidden_representation.add(autoencoder.layers[0])
    hidden_representation.add(autoencoder.layers[1])
    hidden_representation.add(autoencoder.layers[2])

    norm_hid_rep = hidden_representation.predict(x_norm[:3000])
    fraud_hid_rep = hidden_representation.predict(x_fraud)

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
    train()
