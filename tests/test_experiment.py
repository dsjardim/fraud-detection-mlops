import pytest
import pathlib as pl
import os


def test_model_creation():
    assert pl.Path(__file__).parent.joinpath("data", "model.pickle").is_file()
