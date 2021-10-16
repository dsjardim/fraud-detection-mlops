import pytest
import pathlib as pl
import os


def test_model_creation():
    assert pl.Path("/home/runner/work/fraud-detection-mlops/fraud-detection-mlops/data/model.pickle").exists()
