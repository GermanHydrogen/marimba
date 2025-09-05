"""Tests for marimba.core.utils.metadata module."""

import json
from pathlib import Path

import pytest
import yaml

from marimba.core.utils.metadata import (
    MetadataSaverTypes,
    get_saver,
    json_saver,
    yaml_saver,
)


class TestMetadataSaverTypes:
    """Test MetadataSaverTypes enum."""

    @pytest.mark.unit
    def test_metadata_saver_types_values(self):
        """Test MetadataSaverTypes enum values."""
        assert MetadataSaverTypes.json == "json"
        assert MetadataSaverTypes.yaml == "yaml"


class TestJsonSaver:
    """Test json_saver function."""

    @pytest.mark.unit
    def test_json_saver_creates_json_file(self, tmp_path):
        """Test json_saver creates JSON file with correct content."""
        test_data = {"name": "test_dataset", "items": [{"file": "test.jpg", "metadata": {"lat": 37.7749}}]}

        json_saver(tmp_path, "test_metadata", test_data)

        output_file = tmp_path / "test_metadata.json"
        assert output_file.exists()

        with output_file.open("r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data

    @pytest.mark.unit
    def test_json_saver_with_unicode_data(self, tmp_path):
        """Test json_saver handles unicode data correctly."""
        test_data = {"description": "测试数据", "émoji": "🌟"}

        json_saver(tmp_path, "unicode_test", test_data)

        output_file = tmp_path / "unicode_test.json"
        assert output_file.exists()

        with output_file.open("r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data


class TestYamlSaver:
    """Test yaml_saver function."""

    @pytest.mark.unit
    def test_yaml_saver_creates_yaml_file(self, tmp_path):
        """Test yaml_saver creates YAML file with correct content."""
        test_data = {"name": "test_dataset", "items": [{"file": "test.jpg", "metadata": {"lat": 37.7749}}]}

        yaml_saver(tmp_path, "test_metadata", test_data)

        output_file = tmp_path / "test_metadata.yml"
        assert output_file.exists()

        with output_file.open("r", encoding="utf-8") as f:
            loaded_data = yaml.safe_load(f)

        assert loaded_data == test_data

    @pytest.mark.unit
    def test_yaml_saver_with_nested_data(self, tmp_path):
        """Test yaml_saver handles nested data structures."""
        test_data = {
            "dataset": {"name": "complex_test", "nested": {"deep": {"values": [1, 2, 3], "strings": ["a", "b", "c"]}}}
        }

        yaml_saver(tmp_path, "nested_test", test_data)

        output_file = tmp_path / "nested_test.yml"
        assert output_file.exists()

        with output_file.open("r", encoding="utf-8") as f:
            loaded_data = yaml.safe_load(f)

        assert loaded_data == test_data


class TestGetSaver:
    """Test get_saver function."""

    @pytest.mark.unit
    def test_get_saver_json(self):
        """Test get_saver returns json_saver for json type."""
        saver = get_saver(MetadataSaverTypes.json)
        assert saver == json_saver

    @pytest.mark.unit
    def test_get_saver_yaml(self):
        """Test get_saver returns yaml_saver for yaml type."""
        saver = get_saver(MetadataSaverTypes.yaml)
        assert saver == yaml_saver

    @pytest.mark.unit
    def test_get_saver_unknown_type(self):
        """Test get_saver raises ValueError for unknown saver type."""
        # Create a mock invalid saver type
        with pytest.raises(ValueError, match="Unknown saver"):
            get_saver("invalid_type")  # type: ignore[arg-type]

    @pytest.mark.unit
    def test_get_saver_functional_usage(self, tmp_path):
        """Test get_saver returns functional saver."""
        test_data = {"test": "data"}

        # Get json saver and use it
        json_saver_func = get_saver(MetadataSaverTypes.json)
        json_saver_func(tmp_path, "functional_test", test_data)

        # Verify file was created
        assert (tmp_path / "functional_test.json").exists()

        # Get yaml saver and use it
        yaml_saver_func = get_saver(MetadataSaverTypes.yaml)
        yaml_saver_func(tmp_path, "functional_test", test_data)

        # Verify file was created
        assert (tmp_path / "functional_test.yml").exists()
