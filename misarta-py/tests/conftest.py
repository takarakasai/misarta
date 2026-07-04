"""Shared pytest fixtures."""
from pathlib import Path
import pytest
import misarta


URDF_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "misarta-formats" / "tests" / "model" / "urdf" / "test_robot.urdf"
)


@pytest.fixture(scope="session")
def urdf_string() -> str:
    return URDF_PATH.read_text()


@pytest.fixture(scope="session")
def model() -> "misarta.Model":
    return misarta.load_urdf(URDF_PATH)
