import pathlib as pl


def test_model_creation():
    assert pl.Path("data/model.pickle").exists()


def test_metrics_creation():
    assert pl.Path("data/metrics.json").exists()


def test_dataset_availability():
    assert pl.Path("data/creditcard.csv").exists()
