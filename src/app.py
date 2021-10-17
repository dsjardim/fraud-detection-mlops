import os
import pathlib as pl
import pickle

import nest_asyncio
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File

from preprocessing import prepare_dataset

app = FastAPI(title='Credit Card Fraud Detection API')


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to " \
           "https://credit-fraud-detection-mlops.herokuapp.com. "


@app.post("/predict/")
def predict(csv_file: UploadFile = File(...)):
    df = pd.read_csv(csv_file.file)
    data_dict = prepare_dataset(df, False)

    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]

    model_path = "../data/model.pickle"
    clf = pickle.load(open(model_path, 'rb'))

    y_pred = clf.predict(X_test)

    score = clf.score(X_test, y_test)

    return {
        "output_file": "predict_results.csv",
        "clf_score": score
    }
