import reachy_front
import reachy_protocol


def test_package_exposes_version() -> None:
    assert reachy_front.__version__ == "0.1.0"


def test_depends_on_protocol() -> None:
    assert reachy_protocol.__version__
