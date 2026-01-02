from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

TODAY = dt.date(2026, 11, 18)
DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def today() -> dt.date:
    return TODAY


@pytest.fixture(scope="session")
def synthetic_model():
    """A model trained on synthetic data only, so tests never depend on the annotated files."""
    from nl2gql.generate import Generator
    from nl2gql.model import Translator

    return Translator.train(Generator(TODAY, seed=7).examples(4000), TODAY)
