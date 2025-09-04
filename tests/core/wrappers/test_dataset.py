import tempfile
from pathlib import Path
from shutil import rmtree
from typing import Any
from unittest import TestCase, mock

import pytest

from marimba.core.wrappers.dataset import DatasetWrapper


class TestDatasetWrapper(TestCase):
    """
    Class representing a unit test case for the TestDatasetWrapper class.

    Attributes:
        dataset_wrapper (DatasetWrapper): An instance of the DatasetWrapper class.

    Methods:
        setUp: Set up the test case by creating a DatasetWrapper object.
        tearDown: Clean up the test case by deleting the DatasetWrapper object.
        test_check_dataset_mapping: Test the validity of the dataset mapping.
    """

    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        self.dataset_wrapper = DatasetWrapper.create(Path(self.test_dir.name) / "test_dataset")

    def tearDown(self) -> None:
        root_dir = self.dataset_wrapper.root_dir
        del self.dataset_wrapper
        rmtree(root_dir)
        self.test_dir.cleanup()

    def test_check_dataset_mapping(self) -> None:
        """
        Test that checks the validity of the dataset mapping.

        This method tests different scenarios for the dataset mapping and ensures that
        errors are properly logged for invalid mappings while valid mappings pass without issues.

        After parallelisation, the method now logs errors instead of raising exceptions,
        allowing validation to continue and report all issues found.
        """
        # Test that an invalid dataset mapping logs an error
        dataset_mapping: dict[Any, Any] = {
            "test": {Path("nonexistent_file.txt"): (Path("destination.txt"), None, None)},
        }

        with self.assertLogs("DatasetWrapper", level="ERROR") as cm:
            # This should complete without raising an exception but log errors
            self.dataset_wrapper.check_dataset_mapping(dataset_mapping)

            # Check that error was logged for the nonexistent file
            self.assertTrue(any("nonexistent_file.txt does not exist" in msg for msg in cm.output))

        # Test that a valid path mapping does not log errors
        dataset_mapping = {"test": {Path(__file__): (Path("destination.txt"), None, None)}}

        # No errors should be logged for valid mapping
        with self.assertLogs("DatasetWrapper", level="INFO") as cm:
            self.dataset_wrapper.check_dataset_mapping(dataset_mapping)

            # Should log success message
            self.assertTrue(any("Dataset mapping is valid" in msg for msg in cm.output))
