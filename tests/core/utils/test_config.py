import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml

from marimba.core.utils.config import load_config, save_config


class TestLoadConfig:
    """
    A class to test the functionality of the load_config function with different scenarios.
    """

    @pytest.fixture
    def temp_config_setup(self) -> Generator[tuple[Path, dict[str, str]], None, None]:
        """Set up temporary directory and config data for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_data = {"key": "value"}
            yield config_path, config_data

    @pytest.mark.integration
    def test_load_config_with_valid_yaml(self, temp_config_setup: tuple[Path, dict[str, str]]) -> None:
        config_path, config_data = temp_config_setup
        with config_path.open("w", encoding="utf-8") as f:
            f.write("key: value")

        loaded_data = load_config(config_path)
        assert loaded_data == config_data

    @pytest.mark.integration
    def test_load_config_with_invalid_yaml(self, temp_config_setup: tuple[Path, dict[str, str]]) -> None:
        config_path, _ = temp_config_setup
        with config_path.open("w", encoding="utf-8") as f:
            f.write("key: value\ninvalid")

        with pytest.raises(yaml.scanner.ScannerError):
            load_config(config_path)

    @pytest.mark.integration
    def test_load_config_with_nonexistent_file(self, temp_config_setup: tuple[Path, dict[str, str]]) -> None:
        config_path, _ = temp_config_setup
        nonexistent_path = config_path.parent / "nonexistent_config.yaml"
        with pytest.raises(FileNotFoundError):
            load_config(nonexistent_path)

    @pytest.mark.unit
    def test_save_and_load_config(self, temp_config_setup: tuple[Path, dict[str, str]]) -> None:
        """Test saving and then loading a configuration."""
        config_path, _ = temp_config_setup
        config_data = {"name": "test_config", "version": 1, "settings": {"debug": True, "threshold": 0.5}}

        save_config(config_path, config_data)
        loaded_config = load_config(config_path)

        assert loaded_config == config_data
        assert config_path.exists()

    @pytest.mark.unit
    def test_load_config_with_non_dict_content(self, temp_config_setup: tuple[Path, dict[str, str]]) -> None:
        """Test loading configuration with non-dictionary content."""
        config_path, _ = temp_config_setup
        with config_path.open("w", encoding="utf-8") as f:
            f.write("- item1\n- item2")  # This is a list, not dict

        with pytest.raises(TypeError):
            load_config(config_path)
