import pytest
import pathlib as pl


def test_model_creation():
    print(pl.Path.cwd())
    assert pl.Path.cwd().joinpath("data", "model.pickle").is_file()
