"""Tests for marimba.core.schemas.darwin module."""

import pytest

from marimba.core.schemas.base import BaseMetadata
from marimba.core.schemas.darwin import DarwinCoreMetadata


class TestDarwinCoreMetadata:
    """Test DarwinCoreMetadata class."""

    @pytest.mark.unit
    def test_darwin_core_metadata_inheritance(self):
        """Test that DarwinCoreMetadata inherits from BaseMetadata."""
        # Since DarwinCoreMetadata is abstract, test inheritance via class inspection
        assert issubclass(DarwinCoreMetadata, BaseMetadata)

    @pytest.mark.unit
    def test_darwin_core_metadata_class_definition(self):
        """Test that DarwinCoreMetadata is properly defined."""
        assert DarwinCoreMetadata.__name__ == "DarwinCoreMetadata"
        assert DarwinCoreMetadata.__module__ == "marimba.core.schemas.darwin"
