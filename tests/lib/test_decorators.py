"""Tests for marimba.lib.decorators module."""

import pytest

from marimba.lib.decorators import multithreaded


class TestMultithreadedDecorator:
    """Test multithreaded decorator."""

    @pytest.mark.unit
    def test_multithreaded_import(self):
        """Test that multithreaded decorator can be imported."""
        # Simple smoke test to ensure the decorator can be imported
        assert multithreaded is not None
        assert callable(multithreaded)
