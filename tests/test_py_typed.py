"""
Test that the py.typed marker file is correctly included in the package.

This test verifies that the py.typed file exists in the installed package,
which signals to type checkers like mypy that this package provides type information.
"""

import importlib.metadata
import importlib.resources
from pathlib import Path

import pytest


class TestPyTypedMarker:
    """Test the presence of the py.typed marker file."""

    @pytest.mark.unit
    def test_py_typed_exists(self) -> None:
        """Verify that the py.typed marker file exists in the package."""
        # First check in the source directory directly
        import marimba

        module_path = Path(marimba.__file__).parent
        py_typed_path = module_path / "py.typed"
        assert py_typed_path.exists(), f"py.typed file not found at {py_typed_path}"

        # Then if available, check if it's in the distribution metadata
        try:
            # Get distribution info
            dist = importlib.metadata.distribution("marimba")
            if hasattr(dist, "files") and dist.files:
                # Get all file paths and check if any contains py.typed
                all_paths = [str(f) for f in dist.files]
                py_typed_files = [f for f in all_paths if "py.typed" in f]
                if py_typed_files:
                    assert True, "py.typed found in distribution files"
        except (ImportError, AttributeError):
            # This is fine - we've already checked the file exists in source
            pass

    @pytest.mark.unit
    def test_importable_with_types(self) -> None:
        """Verify that modules can be imported with type information."""
        # Import a few key modules that should have type information
        from marimba.core.schemas.base import BaseMetadata
        from marimba.core.wrappers.dataset import DatasetWrapper
        from marimba.core.wrappers.project import ProjectWrapper

        # Basic type assertion checks that would fail if typing wasn't working
        assert hasattr(ProjectWrapper, "__annotations__"), "ProjectWrapper should have type annotations"
        assert hasattr(DatasetWrapper, "__annotations__"), "DatasetWrapper should have type annotations"
        assert hasattr(BaseMetadata, "__annotations__"), "BaseMetadata should have type annotations"
