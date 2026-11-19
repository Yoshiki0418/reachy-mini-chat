import reachy_protocol
import reachy_server


def test_package_exposes_version() -> None:
    assert reachy_server.__version__ == "0.1.0"


def test_depends_on_protocol() -> None:
    assert reachy_protocol.__version__
