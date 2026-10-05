"""Shared pytest fixtures. Adds src/ to sys.path so tests can import the
project modules the same way app.py and the training scripts do."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def base_student():
    """A plausible, valid student record covering every required raw field."""
    return dict(
        school="GP", sex="F", age=16, address="U", famsize="GT3",
        Pstatus="T", Medu=3, Fedu=2, Mjob="services", Fjob="other",
        reason="course", guardian="mother", traveltime=1, studytime=2,
        failures=0, schoolsup="no", famsup="yes", paid="no",
        activities="yes", nursery="yes", higher="yes", internet="yes",
        romantic="no", famrel=4, freetime=3, goout=3, Dalc=1,
        Walc=1, health=4, absences=4, G1=12, G2=13,
    )
