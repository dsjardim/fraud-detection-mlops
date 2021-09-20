import numpy as np
from sklearn import preprocessing
from model import AutoEncoder

np.random.seed(203)


def process_data(data):
    data["Time"] = data["Time"].apply(lambda x: x / 3600 % 24)
    data.head()

    non_fraud = data[data['Class'] == 0].sample(1000)
    fraud = data[data['Class'] == 1]

    df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
    x_raw = df.drop(['Class'], axis=1).values

    x = data.drop(["Class"], axis=1)
    y = data["Class"].values

    x_scale = preprocessing.MinMaxScaler().fit_transform(x.values)

    x_norm = x_scale[y == 0]
    x_fraud = x_scale[y == 1]

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

    return rep_x, rep_y
