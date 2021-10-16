import pytest
import pathlib as pl
import os


def test_model_creation():
    print(pl.Path(__file__).is_file())
    assert pl.Path(__file__).parent.joinpath("data", "model.pickle").is_file()
