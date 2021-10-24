import os


def test_model_creation():
    assert os.path.exists("data/model.pickle")


def test_metrics_creation():
    assert os.path.exists("data/metrics.json")


def test_dataset_availability():
    assert os.path.exists("data/creditcard.csv")
