import pathlib as pl


def test_model_creation():
    assert pl.Path("/home/runner/work/fraud-detection-mlops/fraud-detection-mlops/data/model.pickle").exists()
