"""
Tests for the PipelineWrapper class in marimba.core.wrappers.pipeline.

This module provides comprehensive tests for the pipeline wrapper functionality including:
- Initialization and property access
- Configuration loading and saving
- Pipeline class discovery and instantiation
- Repository management (creation, updates)
- File structure validation
- Error handling scenarios
- Logging configuration
"""

import logging
import shutil
import tempfile
from pathlib import Path

import pytest

from marimba.core.pipeline import BasePipeline
from marimba.core.schemas.base import BaseMetadata
from marimba.core.wrappers.pipeline import PipelineWrapper


class MockTestPipeline(BasePipeline):
    """Mock test pipeline class for testing purposes (renamed to avoid pytest collection)."""

    def __init__(self, root_path, config=None, metadata_class=BaseMetadata, *, dry_run=False):
        super().__init__(root_path, config, metadata_class, dry_run=dry_run)

    @staticmethod
    def get_pipeline_config_schema():
        return {"test_param": "default_value", "test_int": 42}

    @staticmethod
    def get_collection_config_schema():
        return {"collection_param": "default_collection"}

    def _package(self, data_dir, config, **kwargs):
        return {Path("test.txt"): (Path("relative/test.txt"), [], {})}


class TestPipelineWrapperInitialization:
    """Tests for PipelineWrapper initialization and basic properties."""

    @pytest.mark.integration
    def test_init_success(self, tmp_path, mocker):
        """Test successful PipelineWrapper initialization."""
        # Create required file structure
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        config_file = tmp_path / "pipeline.yml"
        config_file.write_text("test: config")

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mock_installer_create = mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")
        mock_installer = mocker.Mock()
        mock_installer_create.return_value = mock_installer

        wrapper = PipelineWrapper(tmp_path, dry_run=False)

        assert wrapper.root_dir == tmp_path
        assert wrapper.repo_dir == tmp_path / "repo"
        assert wrapper.config_path == tmp_path / "pipeline.yml"
        assert wrapper.log_path == tmp_path / f"{tmp_path.name}.log"
        assert wrapper.name == tmp_path.name
        assert not wrapper.dry_run

    @pytest.mark.integration
    def test_init_with_dry_run(self, tmp_path, mocker):
        """Test PipelineWrapper initialization with dry_run=True."""
        # Create required file structure
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        config_file = tmp_path / "pipeline.yml"
        config_file.write_text("test: config")

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(tmp_path, dry_run=True)
        assert wrapper.dry_run

    @pytest.mark.integration
    def test_init_with_string_path(self, tmp_path, mocker):
        """Test PipelineWrapper initialization with string path."""
        # Create required file structure
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        config_file = tmp_path / "pipeline.yml"
        config_file.write_text("test: config")

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(str(tmp_path), dry_run=False)
        assert wrapper.root_dir == tmp_path


class TestPipelineWrapperFileStructureValidation:
    """Tests for file structure validation in PipelineWrapper."""

    @pytest.mark.integration
    def test_missing_root_directory(self, mocker, tmp_path):
        """Test InvalidStructureError when root directory doesn't exist."""
        missing_dir = tmp_path / "nonexistent"

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        with pytest.raises(
            PipelineWrapper.InvalidStructureError,
            match='".*nonexistent" does not exist or is not a directory',
        ):
            PipelineWrapper(missing_dir)

    @pytest.mark.integration
    def test_missing_repo_directory(self, tmp_path, mocker):
        """Test InvalidStructureError when repo directory doesn't exist."""
        config_file = tmp_path / "pipeline.yml"
        config_file.write_text("test: config")
        # repo directory is missing

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        with pytest.raises(
            PipelineWrapper.InvalidStructureError,
            match='".*repo" does not exist or is not a directory',
        ):
            PipelineWrapper(tmp_path)

    @pytest.mark.integration
    def test_missing_config_file(self, tmp_path, mocker):
        """Test InvalidStructureError when config file doesn't exist."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        # config file is missing

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        with pytest.raises(
            PipelineWrapper.InvalidStructureError,
            match='".*pipeline.yml" does not exist or is not a file',
        ):
            PipelineWrapper(tmp_path)

    @pytest.mark.integration
    def test_repo_is_file_not_directory(self, tmp_path, mocker):
        """Test InvalidStructureError when repo exists as file instead of directory."""
        repo_file = tmp_path / "repo"
        repo_file.write_text("not a directory")
        config_file = tmp_path / "pipeline.yml"
        config_file.write_text("test: config")

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        with pytest.raises(
            PipelineWrapper.InvalidStructureError,
            match='".*repo" does not exist or is not a directory',
        ):
            PipelineWrapper(tmp_path)

    @pytest.mark.integration
    def test_config_is_directory_not_file(self, tmp_path, mocker):
        """Test InvalidStructureError when config exists as directory instead of file."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        config_dir = tmp_path / "pipeline.yml"
        config_dir.mkdir()

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        with pytest.raises(
            PipelineWrapper.InvalidStructureError,
            match='".*pipeline.yml" does not exist or is not a file',
        ):
            PipelineWrapper(tmp_path)


class TestPipelineWrapperLogging:
    """Tests for logging setup in PipelineWrapper."""

    @pytest.mark.integration
    def test_setup_logging(self, tmp_path, mocker):
        """Test logging setup with file handler."""
        # Create required file structure
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        config_file = tmp_path / "pipeline.yml"
        config_file.write_text("test: config")

        mock_get_file_handler = mocker.patch("marimba.core.wrappers.pipeline.get_file_handler")
        mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        mock_file_handler = mocker.Mock()
        mock_get_file_handler.return_value = mock_file_handler

        mock_logger = mocker.Mock()
        mock_get_logger.return_value = mock_logger

        wrapper = PipelineWrapper(tmp_path, dry_run=False)

        # Verify file handler was created and added
        mock_get_file_handler.assert_called_once_with(tmp_path, tmp_path.name, False)
        mock_logger.addHandler.assert_called_with(mock_file_handler)

    @pytest.mark.integration
    def test_setup_logging_dry_run(self, tmp_path, mocker):
        """Test logging setup in dry run mode."""
        # Create required file structure
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        config_file = tmp_path / "pipeline.yml"
        config_file.write_text("test: config")

        mock_get_file_handler = mocker.patch("marimba.core.wrappers.pipeline.get_file_handler")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        mock_file_handler = mocker.Mock()
        mock_get_file_handler.return_value = mock_file_handler

        wrapper = PipelineWrapper(tmp_path, dry_run=True)

        # Verify file handler was created with dry_run=True
        mock_get_file_handler.assert_called_once_with(tmp_path, tmp_path.name, True)


class TestPipelineWrapperCreate:
    """Tests for PipelineWrapper.create class method."""

    @pytest.mark.integration
    def test_create_success(self, tmp_path, mocker):
        """Test successful pipeline creation from git repository."""
        # Ensure the target directory doesn't exist
        pipeline_dir = tmp_path / "new_pipeline"

        mock_repo_class = mocker.patch("marimba.core.wrappers.pipeline.Repo")
        mock_save_config = mocker.patch("marimba.core.wrappers.pipeline.save_config")

        mock_repo = mocker.Mock()
        mock_repo_class.clone_from.return_value = mock_repo

        mock_init = mocker.patch.object(PipelineWrapper, "__init__", return_value=None)
        result = PipelineWrapper.create(pipeline_dir, "https://github.com/example/pipeline.git", dry_run=True)

        # Verify directory was created
        assert pipeline_dir.exists()
        assert pipeline_dir.is_dir()

        # Verify git clone was called
        mock_repo_class.clone_from.assert_called_once_with(
            "https://github.com/example/pipeline.git",
            pipeline_dir / "repo",
        )

        # Verify config file was created
        mock_save_config.assert_called_once_with(pipeline_dir / "pipeline.yml", {})

        # Verify PipelineWrapper constructor was called
        mock_init.assert_called_once_with(pipeline_dir, dry_run=True)

    @pytest.mark.integration
    def test_create_directory_exists_error(self, tmp_path):
        """Test FileExistsError when target directory already exists."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        with pytest.raises(FileExistsError, match=f'Pipeline root directory "{existing_dir}" already exists'):
            PipelineWrapper.create(existing_dir, "https://github.com/example/pipeline.git")

    @pytest.mark.integration
    def test_create_with_string_path(self, tmp_path, mocker):
        """Test pipeline creation with string path."""
        pipeline_dir = tmp_path / "string_path_test"

        mock_repo_class = mocker.patch("marimba.core.wrappers.pipeline.Repo")
        mocker.patch.object(PipelineWrapper, "__init__", return_value=None)
        mocker.patch("marimba.core.wrappers.pipeline.save_config")

        PipelineWrapper.create(str(pipeline_dir), "https://github.com/example/pipeline.git")

        # Verify directory was created
        assert pipeline_dir.exists()


class TestPipelineWrapperConfiguration:
    """Tests for configuration loading and saving."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repo_dir = self.temp_dir / "repo"
        self.repo_dir.mkdir()
        self.config_file = self.temp_dir / "pipeline.yml"
        self.config_file.write_text("test: config")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @pytest.mark.integration
    def test_load_config(self, mocker):
        """Test configuration loading."""
        mock_load_config = mocker.patch("marimba.core.wrappers.pipeline.load_config")
        mock_load_config.return_value = {"key": "value"}

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)
        result = wrapper.load_config()

        mock_load_config.assert_called_once_with(wrapper.config_path)
        assert result == {"key": "value"}

    @pytest.mark.integration
    def test_save_config_with_data(self, mocker):
        """Test configuration saving with data."""
        mock_save_config = mocker.patch("marimba.core.wrappers.pipeline.save_config")
        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)
        test_config = {"test": "data"}
        wrapper.save_config(test_config)

        mock_save_config.assert_called_once_with(wrapper.config_path, test_config)

    @pytest.mark.integration
    def test_save_config_with_none(self, mocker):
        """Test configuration saving with None (should not call save_config)."""
        mock_save_config = mocker.patch("marimba.core.wrappers.pipeline.save_config")
        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)
        wrapper.save_config(None)

        mock_save_config.assert_not_called()

    @pytest.mark.integration
    def test_save_config_with_empty_dict(self, mocker):
        """Test configuration saving with empty dictionary (should not call save_config)."""
        mock_save_config = mocker.patch("marimba.core.wrappers.pipeline.save_config")
        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)

        # Empty dict is falsy in Python, so save_config should NOT be called
        wrapper.save_config({})

        mock_save_config.assert_not_called()


class TestPipelineWrapperInstanceManagement:
    """Tests for pipeline instance retrieval."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repo_dir = self.temp_dir / "repo"
        self.repo_dir.mkdir()
        self.config_file = self.temp_dir / "pipeline.yml"
        self.config_file.write_text("test: config")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @pytest.mark.integration
    def test_get_instance_success(self, mocker):
        """Test successful pipeline instance retrieval."""
        mock_load_pipeline_instance = mocker.patch("marimba.core.wrappers.pipeline.load_pipeline_instance")
        mock_pipeline = mocker.Mock(spec=BasePipeline)
        mock_load_pipeline_instance.return_value = mock_pipeline

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir, dry_run=True)
        result = wrapper.get_instance()

        mock_load_pipeline_instance.assert_called_once_with(
            self.temp_dir,
            self.repo_dir,
            self.temp_dir.name,
            self.config_file,
            True,  # dry_run
            log_string_prefix=None,
            allow_empty=False,
        )
        assert result == mock_pipeline

    @pytest.mark.integration
    def test_get_instance_allow_empty_true(self, mocker):
        """Test pipeline instance retrieval with allow_empty=True."""
        mock_load_pipeline_instance = mocker.patch("marimba.core.wrappers.pipeline.load_pipeline_instance")
        mock_load_pipeline_instance.return_value = None

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)
        result = wrapper.get_instance(allow_empty=True)

        mock_load_pipeline_instance.assert_called_once_with(
            self.temp_dir,
            self.repo_dir,
            self.temp_dir.name,
            self.config_file,
            False,  # dry_run
            log_string_prefix=None,
            allow_empty=True,
        )
        assert result is None


class TestPipelineWrapperPipelineClassDiscovery:
    """Tests for pipeline class discovery and caching."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repo_dir = self.temp_dir / "repo"
        self.repo_dir.mkdir()
        self.config_file = self.temp_dir / "pipeline.yml"
        self.config_file.write_text("test: config")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @pytest.mark.integration
    def test_get_pipeline_class_no_files(self, mocker):
        """Test FileNotFoundError when no .pipeline.py files exist."""
        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)

        with pytest.raises(FileNotFoundError, match=f'No pipeline implementation found in "{self.repo_dir}"'):
            wrapper.get_pipeline_class()

    @pytest.mark.integration
    def test_get_pipeline_class_multiple_files(self, mocker):
        """Test FileNotFoundError when multiple .pipeline.py files exist."""
        # Create multiple pipeline files
        pipeline_file1 = self.repo_dir / "first.pipeline.py"
        pipeline_file2 = self.repo_dir / "second.pipeline.py"
        pipeline_file1.write_text("# Pipeline 1")
        pipeline_file2.write_text("# Pipeline 2")

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)

        with pytest.raises(FileNotFoundError, match=f'Multiple pipeline implementations found in "{self.repo_dir}"'):
            wrapper.get_pipeline_class()

    @pytest.mark.integration
    def test_get_pipeline_class_spec_none(self, mocker):
        """Test ImportError when spec_from_file_location returns None."""
        pipeline_file = self.repo_dir / "test.pipeline.py"
        pipeline_file.write_text("# Test pipeline")

        mock_spec_from_file_location = mocker.patch("marimba.core.wrappers.pipeline.spec_from_file_location")
        mock_spec_from_file_location.return_value = None

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)

        with pytest.raises(ImportError, match="Could not load spec for test.pipeline from"):
            wrapper.get_pipeline_class()

    @pytest.mark.integration
    def test_get_pipeline_class_no_loader(self, mocker):
        """Test ImportError when module spec has no loader."""
        pipeline_file = self.repo_dir / "test.pipeline.py"
        pipeline_file.write_text("# Test pipeline")

        mock_spec_from_file_location = mocker.patch("marimba.core.wrappers.pipeline.spec_from_file_location")
        mock_module_from_spec = mocker.patch("marimba.core.wrappers.pipeline.module_from_spec")

        mock_spec = mocker.Mock()
        mock_spec.loader = None
        mock_spec_from_file_location.return_value = mock_spec

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)

        with pytest.raises(ImportError, match="Could not find loader for test.pipeline from"):
            wrapper.get_pipeline_class()

    @pytest.mark.integration
    def test_get_pipeline_class_success(self, mocker):
        """Test successful pipeline class discovery and caching."""
        pipeline_file = self.repo_dir / "test.pipeline.py"
        pipeline_file.write_text("# Test pipeline")

        # Mock the module loading process
        mock_spec_from_file_location = mocker.patch("marimba.core.wrappers.pipeline.spec_from_file_location")
        mock_module_from_spec = mocker.patch("marimba.core.wrappers.pipeline.module_from_spec")

        mock_loader = mocker.Mock()
        mock_spec = mocker.Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file_location.return_value = mock_spec

        # Create a real module-like object with proper __dict__
        mock_module = type("MockModule", (), {})()
        mock_module.__dict__ = {
            "SomeClass": str,  # Not a BasePipeline subclass
            "BasePipeline": BasePipeline,  # This should be ignored
            "TestPipeline": MockTestPipeline,  # This should be found
            "AnotherClass": int,  # Not a BasePipeline subclass
        }
        mock_module_from_spec.return_value = mock_module

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)

        # First call should discover and cache the class
        result = wrapper.get_pipeline_class()
        assert result == MockTestPipeline

        # Second call should return cached result
        result2 = wrapper.get_pipeline_class()
        assert result2 == MockTestPipeline

        # Verify module was only loaded once (caching works)
        mock_spec_from_file_location.assert_called_once()

    @pytest.mark.integration
    def test_get_pipeline_class_no_valid_class(self, mocker):
        """Test that None is returned when no valid pipeline class is found."""
        pipeline_file = self.repo_dir / "test.pipeline.py"
        pipeline_file.write_text("# Test pipeline")

        # Mock the module loading process
        mock_spec_from_file_location = mocker.patch("marimba.core.wrappers.pipeline.spec_from_file_location")
        mock_module_from_spec = mocker.patch("marimba.core.wrappers.pipeline.module_from_spec")

        mock_loader = mocker.Mock()
        mock_spec = mocker.Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file_location.return_value = mock_spec

        # Create a real module-like object with proper __dict__
        mock_module = type("MockModule", (), {})()
        mock_module.__dict__ = {
            "SomeClass": str,  # Not a BasePipeline subclass
            "BasePipeline": BasePipeline,  # This should be ignored (it's the base class itself)
            "AnotherClass": int,  # Not a BasePipeline subclass
        }
        mock_module_from_spec.return_value = mock_module

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)

        result = wrapper.get_pipeline_class()
        assert result is None


class TestPipelineWrapperPipelineConfigPrompt:
    """Tests for pipeline configuration prompting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repo_dir = self.temp_dir / "repo"
        self.repo_dir.mkdir()
        self.config_file = self.temp_dir / "pipeline.yml"
        self.config_file.write_text("test: config")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @pytest.mark.integration
    def test_prompt_pipeline_config_success(self, mocker):
        """Test successful pipeline configuration prompting."""
        # Set up mocks
        mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
        mock_prompt_schema = mocker.patch("marimba.core.wrappers.pipeline.prompt_schema")

        # Mock the pipeline instance
        mock_pipeline = mocker.Mock()
        mock_pipeline.get_pipeline_config_schema.return_value = {"param1": "default1", "param2": 42}

        mock_prompt_schema.return_value = {"param1": "user_value"}

        mock_logger = mocker.Mock()
        mock_get_logger.return_value = mock_logger

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mock_get_instance = mocker.patch.object(PipelineWrapper, "get_instance")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        mock_get_instance.return_value = mock_pipeline

        wrapper = PipelineWrapper(self.temp_dir)

        result = wrapper.prompt_pipeline_config()

        # Verify pipeline instance was retrieved
        mock_get_instance.assert_called_once_with(allow_empty=False)

        # Verify schema was prompted
        mock_prompt_schema.assert_called_once_with({"param1": "default1", "param2": 42})

        # Verify result
        assert result == {"param1": "user_value"}

    @pytest.mark.integration
    def test_prompt_pipeline_config_with_existing_config(self, mocker):
        """Test pipeline configuration prompting with existing config."""
        mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
        mock_prompt_schema = mocker.patch("marimba.core.wrappers.pipeline.prompt_schema")

        mock_pipeline = mocker.Mock()
        mock_pipeline.get_pipeline_config_schema.return_value = {
            "param1": "default1",
            "param2": 42,
            "param3": "default3",
        }

        mock_prompt_schema.return_value = {"param2": 100}

        mock_logger = mocker.Mock()
        mock_get_logger.return_value = mock_logger

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mock_get_instance = mocker.patch.object(PipelineWrapper, "get_instance")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        mock_get_instance.return_value = mock_pipeline

        wrapper = PipelineWrapper(self.temp_dir)

        existing_config = {"param1": "existing_value"}
        result = wrapper.prompt_pipeline_config(config=existing_config)

        # Verify that existing param1 was pre-populated and not prompted
        mock_prompt_schema.assert_called_once_with({"param2": 42, "param3": "default3"})

        # Verify result contains both existing and prompted values
        assert result == {"param1": "existing_value", "param2": 100}

    @pytest.mark.integration
    def test_prompt_pipeline_config_empty_pipeline(self, mocker):
        """Test pipeline configuration prompting with empty pipeline."""
        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mock_get_instance = mocker.patch.object(PipelineWrapper, "get_instance")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        mock_get_instance.return_value = None

        wrapper = PipelineWrapper(self.temp_dir)

        result = wrapper.prompt_pipeline_config(allow_empty=True)

        mock_get_instance.assert_called_once_with(allow_empty=True)
        assert result is None

    @pytest.mark.integration
    def test_prompt_pipeline_config_no_additional_prompting(self, mocker):
        """Test pipeline configuration when no additional prompting is needed."""
        mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
        mock_prompt_schema = mocker.patch("marimba.core.wrappers.pipeline.prompt_schema")

        mock_pipeline = mocker.Mock()
        mock_pipeline.get_pipeline_config_schema.return_value = {"param1": "default1"}

        mock_logger = mocker.Mock()
        mock_get_logger.return_value = mock_logger

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mock_get_instance = mocker.patch.object(PipelineWrapper, "get_instance")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        mock_get_instance.return_value = mock_pipeline

        wrapper = PipelineWrapper(self.temp_dir)

        existing_config = {"param1": "existing_value"}
        result = wrapper.prompt_pipeline_config(config=existing_config)

        # No prompting should occur since all parameters are satisfied
        mock_prompt_schema.assert_not_called()

        # Result should contain the existing config
        assert result == {"param1": "existing_value"}

    @pytest.mark.integration
    def test_prompt_pipeline_config_with_project_logger(self, mocker):
        """Test pipeline configuration prompting with custom project logger."""
        mock_prompt_schema = mocker.patch("marimba.core.wrappers.pipeline.prompt_schema")

        mock_pipeline = mocker.Mock()
        mock_pipeline.get_pipeline_config_schema.return_value = {}

        mock_project_logger = mocker.Mock(spec=logging.Logger)

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mock_get_instance = mocker.patch.object(PipelineWrapper, "get_instance")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        mock_get_instance.return_value = mock_pipeline

        wrapper = PipelineWrapper(self.temp_dir)

        result = wrapper.prompt_pipeline_config(project_logger=mock_project_logger)

        # Verify project logger was used instead of pipeline logger
        mock_project_logger.info.assert_called_once_with("Provided pipeline config={}")

    @pytest.mark.integration
    def test_prompt_pipeline_config_uses_pipeline_logger_by_default(self, mocker):
        """Test that pipeline logger is used when no project logger is provided."""
        mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
        mock_prompt_schema = mocker.patch("marimba.core.wrappers.pipeline.prompt_schema")

        mock_pipeline = mocker.Mock()
        mock_pipeline.get_pipeline_config_schema.return_value = {}

        mock_logger = mocker.Mock()
        mock_get_logger.return_value = mock_logger

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mock_get_instance = mocker.patch.object(PipelineWrapper, "get_instance")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        mock_get_instance.return_value = mock_pipeline

        wrapper = PipelineWrapper(self.temp_dir)

        result = wrapper.prompt_pipeline_config()

        # Verify pipeline's logger was used (mock_logger through get_logger)
        mock_logger.info.assert_called_once_with("Provided pipeline config={}")


class TestPipelineWrapperRepositoryOperations:
    """Tests for repository update and installation operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repo_dir = self.temp_dir / "repo"
        self.repo_dir.mkdir()
        self.config_file = self.temp_dir / "pipeline.yml"
        self.config_file.write_text("test: config")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @pytest.mark.integration
    def test_update_success(self, mocker):
        """Test successful repository update."""
        mock_repo_class = mocker.patch("marimba.core.wrappers.pipeline.Repo")
        mock_repo = mocker.Mock()
        mock_origin = mocker.Mock()
        mock_repo.remotes.origin = mock_origin
        mock_repo_class.return_value = mock_repo

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)

        wrapper.update()

        mock_repo_class.assert_called_once_with(self.repo_dir)
        mock_origin.pull.assert_called_once()

    @pytest.mark.integration
    def test_install_success(self, mocker):
        """Test successful pipeline dependency installation."""
        mock_installer = mocker.Mock()

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mock_installer_create = mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        mock_installer_create.return_value = mock_installer

        wrapper = PipelineWrapper(self.temp_dir)
        wrapper.install()

        mock_installer.assert_called_once()


class TestPipelineWrapperErrorHandling:
    """Tests for error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repo_dir = self.temp_dir / "repo"
        self.repo_dir.mkdir()
        self.config_file = self.temp_dir / "pipeline.yml"
        self.config_file.write_text("test: config")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @pytest.mark.integration
    def test_update_git_error(self, mocker):
        """Test repository update with git error."""
        from git.exc import GitError

        mock_repo_class = mocker.patch("marimba.core.wrappers.pipeline.Repo")
        mock_repo_class.side_effect = GitError("Git operation failed")

        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")

        wrapper = PipelineWrapper(self.temp_dir)

        with pytest.raises(GitError):
            wrapper.update()

    @pytest.mark.integration
    def test_create_git_clone_error(self, mocker):
        """Test pipeline creation with git clone error."""
        from git.exc import GitError

        mock_save_config = mocker.patch("marimba.core.wrappers.pipeline.save_config")
        mock_repo_class = mocker.patch("marimba.core.wrappers.pipeline.Repo")

        pipeline_dir = self.temp_dir / "new_pipeline"
        mock_repo_class.clone_from.side_effect = GitError("Clone failed")

        with pytest.raises(GitError):
            PipelineWrapper.create(pipeline_dir, "https://github.com/example/pipeline.git")

        # Directory should still exist but be empty/incomplete
        assert pipeline_dir.exists()
        mock_save_config.assert_not_called()  # Config save shouldn't happen if clone fails
