import tempfile
from pathlib import Path
from unittest import TestCase

import pytest
import yaml

from marimba.core.utils.config import load_config, save_config


class TestLoadConfig(TestCase):
    """
    A class to test the functionality of the load_config function with different scenarios.

    Attributes:
        config_path (Path): The path to the test config file.
        config_data (dict): The expected configuration data.

    Methods:
        setUp() -> None:
            Set up the necessary attributes for testing.

        tearDown() -> None:
            Clean up the test environment after testing.

        test_load_config_with_valid_yaml() -> None:
            Test the load_config function with a valid YAML file.
            Assert that the loaded configuration data is equal to the expected data.

        test_load_config_with_invalid_yaml() -> None:
            Test the load_config function with an invalid YAML file.
            Assert that a yaml.scanner.ScannerError is raised.

        test_load_config_with_nonexistent_file() -> None:
            Test the load_config function with a nonexistent file.
            Assert that a FileNotFoundError is raised.
    """

    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.test_dir.name) / "test_config.yaml"
        self.config_data = {"key": "value"}

    def tearDown(self) -> None:
        self.test_dir.cleanup()

    @pytest.mark.integration
    def test_load_config_with_valid_yaml(self) -> None:
        with self.config_path.open("w", encoding="utf-8") as f:
            f.write("key: value")

        config_data = load_config(self.config_path)
        self.assertEqual(config_data, self.config_data)

    @pytest.mark.integration
    def test_load_config_with_invalid_yaml(self) -> None:
        with self.config_path.open("w", encoding="utf-8") as f:
            f.write("key: value\ninvalid")

        with self.assertRaises(yaml.scanner.ScannerError):
            load_config(self.config_path)

    @pytest.mark.integration
    def test_load_config_with_nonexistent_file(self) -> None:
        nonexistent_path = Path(self.test_dir.name) / "nonexistent_config.yaml"
        with self.assertRaises(FileNotFoundError):
            load_config(nonexistent_path)

    @pytest.mark.unit
    def test_save_and_load_config(self) -> None:
        """Test saving and then loading a configuration."""
        config_data = {"name": "test_config", "version": 1, "settings": {"debug": True, "threshold": 0.5}}

        save_config(self.config_path, config_data)
        loaded_config = load_config(self.config_path)

        self.assertEqual(loaded_config, config_data)
        self.assertTrue(self.config_path.exists())

    @pytest.mark.unit
    def test_load_config_with_non_dict_content(self) -> None:
        """Test loading configuration with non-dictionary content."""
        with self.config_path.open("w", encoding="utf-8") as f:
            f.write("- item1\n- item2")  # This is a list, not dict

        with self.assertRaises(TypeError):
            load_config(self.config_path)
