import pytest
import pathlib as pl


def test_model_creation():
    assert pl.Path("data/model.pickle").is_file()
