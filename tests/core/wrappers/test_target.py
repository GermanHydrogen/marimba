"""
Unit tests for DistributionTargetWrapper.

Tests the functionality of the DistributionTargetWrapper class including:
- Loading and validating configuration files
- Creating new targets
- Getting instances of distribution targets
- Error handling for invalid configurations
"""

import tempfile
from pathlib import Path

import pytest

from marimba.core.wrappers.target import DistributionTargetWrapper


@pytest.mark.unit
class TestDistributionTargetWrapper:
    """Test DistributionTargetWrapper functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def valid_s3_config(self):
        """Return a valid S3 configuration."""
        return {
            "type": "s3",
            "config": {
                "bucket_name": "test-bucket",
                "endpoint_url": "https://s3.amazonaws.com",
                "access_key_id": "test-key",
                "secret_access_key": "test-secret",
            },
        }

    @pytest.fixture
    def valid_dap_config(self):
        """Return a valid DAP configuration."""
        return {
            "type": "dap",
            "config": {
                "endpoint_url": "https://test.dap.server.com",
                "access_key": "test_user",
                "secret_access_key": "test_password",
                "remote_directory": "bucket/datasets",
            },
        }

    def test_init_with_valid_config(self, temp_dir, valid_s3_config):
        """Test initialization with a valid configuration file."""
        config_path = temp_dir / "target.yml"

        # Manually write the config file
        import yaml

        config_path.write_text(yaml.dump(valid_s3_config))

        wrapper = DistributionTargetWrapper(config_path)

        assert wrapper.config_path == config_path
        assert wrapper.config == valid_s3_config

    def test_init_with_nonexistent_file(self, temp_dir):
        """Test initialization with non-existent configuration file."""
        config_path = temp_dir / "nonexistent.yml"

        with pytest.raises(FileNotFoundError):
            DistributionTargetWrapper(config_path)

    def test_init_with_invalid_config_no_type(self, temp_dir):
        """Test initialization with configuration missing type."""
        config_path = temp_dir / "target.yml"
        invalid_config = {"config": {"bucket": "test"}}

        import yaml

        config_path.write_text(yaml.dump(invalid_config))

        with pytest.raises(DistributionTargetWrapper.InvalidConfigError, match="must specify a 'type'"):
            DistributionTargetWrapper(config_path)

    def test_init_with_invalid_config_no_config(self, temp_dir):
        """Test initialization with configuration missing config section."""
        config_path = temp_dir / "target.yml"
        invalid_config = {"type": "s3"}

        import yaml

        config_path.write_text(yaml.dump(invalid_config))

        with pytest.raises(DistributionTargetWrapper.InvalidConfigError, match="must specify a 'config'"):
            DistributionTargetWrapper(config_path)

    def test_init_with_invalid_target_type(self, temp_dir):
        """Test initialization with invalid target type."""
        config_path = temp_dir / "target.yml"
        invalid_config = {"type": "invalid", "config": {"test": "value"}}

        import yaml

        config_path.write_text(yaml.dump(invalid_config))

        with pytest.raises(DistributionTargetWrapper.InvalidConfigError, match="Invalid distribution target type"):
            DistributionTargetWrapper(config_path)

    def test_create_s3_target(self, temp_dir):
        """Test creating an S3 target."""
        config_path = temp_dir / "target.yml"
        target_type = "s3"
        target_args = {
            "bucket_name": "test-bucket",
            "endpoint_url": "https://s3.amazonaws.com",
            "access_key_id": "test-key",
            "secret_access_key": "test-secret",
        }

        wrapper = DistributionTargetWrapper.create(config_path, target_type, target_args)

        assert wrapper.config_path == config_path
        assert wrapper.config["type"] == target_type
        assert wrapper.config["config"] == target_args
        assert config_path.exists()

    def test_create_dap_target(self, temp_dir):
        """Test creating a DAP target."""
        config_path = temp_dir / "target.yml"
        target_type = "dap"
        target_args = {
            "endpoint_url": "https://test.dap.server.com",
            "access_key": "test_user",
            "secret_access_key": "test_password",
            "remote_directory": "bucket/datasets",
        }

        wrapper = DistributionTargetWrapper.create(config_path, target_type, target_args)

        assert wrapper.config_path == config_path
        assert wrapper.config["type"] == target_type
        assert wrapper.config["config"] == target_args
        assert config_path.exists()

    def test_create_existing_file_error(self, temp_dir):
        """Test creating target when file already exists."""
        config_path = temp_dir / "target.yml"
        config_path.write_text("existing file")

        with pytest.raises(FileExistsError):
            DistributionTargetWrapper.create(config_path, "s3", {"test": "value"})

    def test_get_instance_s3(self, temp_dir, valid_s3_config):
        """Test getting an S3 distribution target instance."""
        config_path = temp_dir / "target.yml"

        import yaml

        config_path.write_text(yaml.dump(valid_s3_config))

        wrapper = DistributionTargetWrapper(config_path)
        instance = wrapper.get_instance()

        assert instance is not None
        assert instance.__class__.__name__ == "S3DistributionTarget"

    def test_get_instance_dap(self, temp_dir, valid_dap_config):
        """Test getting a DAP distribution target instance."""
        config_path = temp_dir / "target.yml"

        import yaml

        config_path.write_text(yaml.dump(valid_dap_config))

        wrapper = DistributionTargetWrapper(config_path)
        instance = wrapper.get_instance()

        assert instance is not None
        assert instance.__class__.__name__ == "CSIRODapDistributionTarget"

    def test_get_instance_invalid_config(self, temp_dir):
        """Test getting instance with malformed config returns None."""
        config_path = temp_dir / "target.yml"

        # Create wrapper with valid config, then modify config to be invalid
        valid_config = {"type": "s3", "config": {"bucket": "test"}}
        import yaml

        config_path.write_text(yaml.dump(valid_config))

        wrapper = DistributionTargetWrapper(config_path)
        # Modify the config to break get_instance
        wrapper._config["type"] = None

        instance = wrapper.get_instance()
        assert instance is None

    def test_get_instance_unknown_type(self, temp_dir):
        """Test getting instance with unknown type returns None."""
        config_path = temp_dir / "target.yml"

        # Create wrapper, then modify config to have unknown type
        valid_config = {"type": "s3", "config": {"bucket": "test"}}
        import yaml

        config_path.write_text(yaml.dump(valid_config))

        wrapper = DistributionTargetWrapper(config_path)
        wrapper._config["type"] = "unknown"

        instance = wrapper.get_instance()
        assert instance is None

    def test_prompt_target_s3(self, mocker):
        """Test interactive prompting for S3 target creation."""
        mock_ask = mocker.patch("rich.prompt.Prompt.ask")
        # Mock the user inputs
        mock_ask.side_effect = [
            "s3",  # target type
            "test-bucket",  # bucket_name
            "https://s3.amazonaws.com",  # endpoint_url
            "test-key",  # access_key_id
            "test-secret",  # secret_access_key
            "",  # base_prefix (default empty)
        ]

        target_type, target_args = DistributionTargetWrapper.prompt_target()

        assert target_type == "s3"
        assert "bucket_name" in target_args
        assert target_args["bucket_name"] == "test-bucket"

    def test_prompt_target_dap(self, mocker):
        """Test interactive prompting for DAP target creation."""
        mock_ask = mocker.patch("rich.prompt.Prompt.ask")
        # Mock the user inputs
        mock_ask.side_effect = [
            "dap",  # target type
            "https://test.server.com",  # endpoint_url
            "access_key",  # access_key
            "secret_key",  # secret_access_key
            "bucket/data",  # remote_directory
        ]

        target_type, target_args = DistributionTargetWrapper.prompt_target()

        assert target_type == "dap"
        assert "endpoint_url" in target_args
        assert target_args["endpoint_url"] == "https://test.server.com"

    def test_class_map_contains_expected_types(self):
        """Test that the CLASS_MAP contains expected target types."""
        assert "s3" in DistributionTargetWrapper.CLASS_MAP
        assert "dap" in DistributionTargetWrapper.CLASS_MAP

        # Verify the classes are correct
        from marimba.core.distribution.s3 import S3DistributionTarget
        from marimba.core.distribution.dap import CSIRODapDistributionTarget

        assert DistributionTargetWrapper.CLASS_MAP["s3"] == S3DistributionTarget
        assert DistributionTargetWrapper.CLASS_MAP["dap"] == CSIRODapDistributionTarget

    def test_config_path_property(self, temp_dir, valid_s3_config):
        """Test config_path property returns correct path."""
        config_path = temp_dir / "target.yml"

        import yaml

        config_path.write_text(yaml.dump(valid_s3_config))

        wrapper = DistributionTargetWrapper(config_path)
        assert wrapper.config_path == config_path

    def test_config_property(self, temp_dir, valid_s3_config):
        """Test config property returns correct configuration."""
        config_path = temp_dir / "target.yml"

        import yaml

        config_path.write_text(yaml.dump(valid_s3_config))

        wrapper = DistributionTargetWrapper(config_path)
        assert wrapper.config == valid_s3_config

    def test_load_config_called_during_init(self, mocker, temp_dir, valid_s3_config):
        """Test that _load_config is called during initialization."""
        config_path = temp_dir / "target.yml"

        import yaml

        config_path.write_text(yaml.dump(valid_s3_config))

        mock_load = mocker.patch.object(DistributionTargetWrapper, "_load_config")
        mock_load.return_value = None
        wrapper = DistributionTargetWrapper.__new__(DistributionTargetWrapper)
        wrapper._config_path = config_path
        wrapper._config = {}
        wrapper._load_config()

        mock_load.assert_called_once()

    def test_check_config_called_during_init(self, mocker, temp_dir, valid_s3_config):
        """Test that _check_config is called during initialization."""
        config_path = temp_dir / "target.yml"

        import yaml

        config_path.write_text(yaml.dump(valid_s3_config))

        mock_check = mocker.patch.object(DistributionTargetWrapper, "_check_config")
        mock_check.return_value = None
        wrapper = DistributionTargetWrapper.__new__(DistributionTargetWrapper)
        wrapper._config_path = config_path
        wrapper._config = valid_s3_config
        wrapper._check_config()

        mock_check.assert_called_once()


@pytest.mark.unit
class TestDistributionTargetWrapperPromptEdgeCases:
    """Test edge cases for the prompt_target method."""

    def test_prompt_target_invalid_class_map_entry(self, mocker):
        """Test prompt_target with invalid class in CLASS_MAP."""
        # Temporarily modify CLASS_MAP to include invalid entry
        original_map = DistributionTargetWrapper.CLASS_MAP.copy()
        DistributionTargetWrapper.CLASS_MAP["invalid"] = str  # type: ignore[assignment]

        try:
            mocker.patch("rich.prompt.Prompt.ask", return_value="invalid")
            with pytest.raises(TypeError, match="__init__ of target class invalid is not a method"):
                DistributionTargetWrapper.prompt_target()
        finally:
            # Restore original CLASS_MAP
            DistributionTargetWrapper.CLASS_MAP.clear()
            DistributionTargetWrapper.CLASS_MAP.update(original_map)

    def test_prompt_target_missing_init_method(self, mocker):
        """Test prompt_target with class missing __init__ method."""

        # Create a class that doesn't inherit from object (rare case)
        # In Python 3, all classes have __init__ by default, so this test
        # is more of a theoretical edge case
        class BadClass:
            def __new__(cls):
                return object.__new__(cls)

        # The function checks isinstance(target_class.__init__, FunctionType)
        # which will be False for built-in methods
        original_map = DistributionTargetWrapper.CLASS_MAP.copy()
        DistributionTargetWrapper.CLASS_MAP["bad_class"] = BadClass

        try:
            mocker.patch("rich.prompt.Prompt.ask", return_value="bad_class")
            with pytest.raises(TypeError, match="is not a method"):
                DistributionTargetWrapper.prompt_target()
        finally:
            DistributionTargetWrapper.CLASS_MAP.clear()
            DistributionTargetWrapper.CLASS_MAP.update(original_map)

    def test_prompt_target_nonexistent_type(self, mocker):
        """Test prompt_target with nonexistent target type."""
        mocker.patch("rich.prompt.Prompt.ask", return_value="nonexistent")
        with pytest.raises(ValueError, match="No target class found for type"):
            DistributionTargetWrapper.prompt_target()
