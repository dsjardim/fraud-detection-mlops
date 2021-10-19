import pickle

import pandas as pd
import pathlib as pl
from fastapi import FastAPI, UploadFile, File

from src.preprocessing import prepare_dataset
from src.data_utils import download_data_from_s3

app = FastAPI(title='Credit Card Fraud Detection API')


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to " \
           "https://credit-fraud-detection-mlops.herokuapp.com/docs. "


@app.post("/predict/")
def predict(csv_file: UploadFile = File(...)):
    df = pd.read_csv(csv_file.file)
    data_dict = prepare_dataset(df, False)

    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]

    model_path = "model.pickle"
    if pl.Path(model_path).exists():
        clf = pickle.load(open(model_path, 'rb'))
    else:
        download_data_from_s3('credit-fraud-mlops-artifacts', 'model.pickle', model_path)
        clf = pickle.load(open(model_path, 'rb'))

    print(clf)

    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test).tolist()

    return {
        "clf_score": score,
        "predictions": y_pred
    }
