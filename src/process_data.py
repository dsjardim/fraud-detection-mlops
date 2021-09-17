from sklearn import preprocessing


def process_data(data):
    data["Time"] = data["Time"].apply(lambda x: x / 3600 % 24)
    data.head()

    vc = data['Class'].value_counts().to_frame().reset_index()
    vc['percent'] = vc["Class"].apply(lambda x: round(100 * float(x) / len(data), 2))
    vc = vc.rename(columns={"index": "Target", "Class": "Count"})

    non_fraud = data[data['Class'] == 0].sample(1000)
    fraud = data[data['Class'] == 1]

    df = non_fraud.append(fraud).sample(frac=1).reset_index(drop=True)
    x_raw = df.drop(['Class'], axis=1).values
    y_raw = df["Class"].values

    x = data.drop(["Class"], axis=1)
    y = data["Class"].values

    x_scale = preprocessing.MinMaxScaler().fit_transform(x.values)
    return x_raw, y_raw, x_scale[y == 0], x_scale[y == 1]
