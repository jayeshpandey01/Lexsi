"""Basic test to verify project scaffolding and package imports."""

import src


def test_package_metadata() -> None:
    """Verify package import and version presence."""
    assert src.__version__ == "0.1.0"
