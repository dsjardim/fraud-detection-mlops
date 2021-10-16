import pytest
import pathlib as pl
import os


def test_model_creation():
    print(os.getcwd())
    print(__file__)
    assert pl.Path("data/model.pickle").is_file()
