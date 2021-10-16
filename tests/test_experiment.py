import pytest
import pathlib as pl
import os


def test_model_creation1():
    assert pl.Path("/home/runner/work/fraud-detection-mlops/fraud-detection-mlops/data/model.pickle").is_file()


def test_model_creation2():
    assert pl.Path("/home/runner/work/fraud-detection-mlops/data/model.pickle").is_file()
