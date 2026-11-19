import reachy_protocol


def test_package_exposes_version() -> None:
    assert reachy_protocol.__version__ == "0.1.0"
