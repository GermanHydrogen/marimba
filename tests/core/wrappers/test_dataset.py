"""Tests for marimba.core.wrappers.dataset module."""

import tempfile
from pathlib import Path
from typing import Any

import pytest

from marimba.core.wrappers.dataset import DatasetWrapper


class TestDatasetWrapper:
    """Test DatasetWrapper functionality."""

    @pytest.fixture(autouse=True)
    def setup_dataset_wrapper(self, tmp_path):
        """Set up a DatasetWrapper instance for testing."""
        self.dataset_wrapper = DatasetWrapper.create(tmp_path / "test_dataset")
        yield
        # Cleanup
        self.dataset_wrapper.close()

    @pytest.mark.integration
    def test_check_dataset_mapping(self) -> None:
        """
        Test that checks the validity of the dataset mapping.

        This method tests different scenarios for the dataset mapping and ensures that they either raise an error or
        pass without any errors. After fixing the parallelisation issue, exceptions are now properly raised again.
        """
        # Test that an invalid dataset mapping raises an error
        dataset_mapping: dict[Any, Any] = {
            "test": {Path("nonexistent_file.txt"): (Path("destination.txt"), None, None)},
        }
        with pytest.raises(DatasetWrapper.InvalidDatasetMappingError):
            self.dataset_wrapper.check_dataset_mapping(dataset_mapping)

        # Test that a valid path mapping does not raise an error
        dataset_mapping = {"test": {Path(__file__): (Path("destination.txt"), None, None)}}
        # This should complete without raising an exception
        self.dataset_wrapper.check_dataset_mapping(dataset_mapping)

        # Test that a path mapping with duplicate source paths raises an error
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file1 = temp_path / "file1.txt"
            file2 = temp_path / "file2.txt"
            file1.touch()
            file2.touch()

            # Create symlink that resolves to same file as file1
            link_file = temp_path / "some_dir" / "link.txt"
            link_file.parent.mkdir(exist_ok=True)
            link_file.symlink_to(file1.absolute())

            dataset_mapping = {
                "test": {
                    file1: (Path("destination1.txt"), None, None),
                    file2: (Path("destination2.txt"), None, None),
                    link_file: (Path("destination3.txt"), None, None),  # This should conflict with file1
                },
            }
            with pytest.raises(DatasetWrapper.InvalidDatasetMappingError):
                self.dataset_wrapper.check_dataset_mapping(dataset_mapping)

        # Test that a path mapping with absolute destination paths raises an error
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "file.txt"
            temp_file.touch()

            dataset_mapping = {"test": {temp_file: (Path("/tmp/absolute_destination.txt"), None, None)}}
            with pytest.raises(DatasetWrapper.InvalidDatasetMappingError):
                self.dataset_wrapper.check_dataset_mapping(dataset_mapping)

        # Test that a path mapping with colliding destination paths raises an error
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file1 = temp_path / "file1.txt"
            file2 = temp_path / "file2.txt"
            file1.touch()
            file2.touch()

            dataset_mapping = {
                "test": {
                    file1: (Path("destination.txt"), None, None),
                    file2: (Path("destination.txt"), None, None),  # Same destination - should conflict
                },
            }
            with pytest.raises(DatasetWrapper.InvalidDatasetMappingError):
                self.dataset_wrapper.check_dataset_mapping(dataset_mapping)

    @pytest.mark.integration
    def test_check_dataset_mapping_comprehensive(self) -> None:
        """
        Additional comprehensive tests for dataset mapping validation edge cases.
        """
        # Test multiple validation failures at once - should aggregate all errors
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            existing_file = temp_path / "existing.txt"
            existing_file.touch()

            dataset_mapping: dict[str, dict[Path, tuple[Path, list[Any] | None, dict[str, Any] | None]]] = {
                "test": {
                    # Multiple issues in one mapping
                    Path("nonexistent1.txt"): (Path("dest1.txt"), None, None),  # Missing source
                    Path("nonexistent2.txt"): (Path("dest2.txt"), None, None),  # Missing source
                    existing_file: (Path("/tmp/absolute_dest.txt"), None, None),  # Absolute destination
                    # Note: Can't test destination collisions in same mapping due to dict key constraints
                },
            }

            with pytest.raises(DatasetWrapper.InvalidDatasetMappingError) as exc_info:
                self.dataset_wrapper.check_dataset_mapping(dataset_mapping)

            # Should contain multiple error messages
            error_message = str(exc_info.value)
            assert "nonexistent1.txt does not exist" in error_message
            assert "nonexistent2.txt does not exist" in error_message
            assert "must be relative" in error_message

        # Test empty dataset mapping - should be valid
        empty_mapping: dict[str, dict[Path, tuple[Path, list[Any] | None, dict[str, Any] | None]]] = {}
        # Should not raise an exception
        self.dataset_wrapper.check_dataset_mapping(empty_mapping)

        # Test mapping with simple destination paths - should be valid
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "file.txt"
            temp_file.touch()

            simple_mapping: dict[str, dict[Path, tuple[Path, list[Any] | None, dict[str, Any] | None]]] = {
                "test": {
                    temp_file: (Path("simple_destination.txt"), None, None),  # Valid destination
                }
            }
            # Should not raise an exception
            self.dataset_wrapper.check_dataset_mapping(simple_mapping)

        # Test symlinks that resolve to the same file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create target file
            target_file = temp_path / "target.txt"
            target_file.touch()

            # Create symlink to target file
            link_file = temp_path / "link_to_target.txt"
            link_file.symlink_to(target_file)

            symlink_mapping: dict[str, dict[Path, tuple[Path, list[Any] | None, dict[str, Any] | None]]] = {
                "test": {
                    target_file: (Path("dest1.txt"), None, None),
                    link_file: (Path("dest2.txt"), None, None),  # Same resolved source
                },
            }
            with pytest.raises(DatasetWrapper.InvalidDatasetMappingError) as exc_info:
                self.dataset_wrapper.check_dataset_mapping(symlink_mapping)

            error_message = str(exc_info.value)
            assert "both resolve to" in error_message

    @pytest.mark.integration
    def test_allow_destination_collisions_flag(self) -> None:
        """Test that allow_destination_collisions flag allows collisions with warning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file1 = temp_path / "file1.txt"
            file2 = temp_path / "file2.txt"
            file1.touch()
            file2.touch()

            # Same destination path but different sources
            collision_mapping: dict[str, dict[Path, tuple[Path, list[Any] | None, dict[str, Any] | None]]] = {
                "test": {
                    file1: (Path("same_destination.txt"), None, None),
                    file2: (Path("same_destination.txt"), None, None),  # Same destination - collision
                },
            }

            # Should fail without the flag
            with pytest.raises(DatasetWrapper.InvalidDatasetMappingError):
                self.dataset_wrapper.check_dataset_mapping(collision_mapping)

            # Should succeed with the flag (no exception raised)
            self.dataset_wrapper.check_dataset_mapping(collision_mapping, allow_destination_collisions=True)
