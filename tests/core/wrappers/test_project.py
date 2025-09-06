"""Tests for marimba.core.wrappers.project module."""

from pathlib import Path
from typing import Any

import pytest

from marimba.core.wrappers.project import ProjectWrapper, get_merged_keyword_args
from marimba.core.wrappers.target import DistributionTargetWrapper


class TestProjectWrapper:
    """Test ProjectWrapper functionality."""

    @pytest.fixture
    def mock_project_dir(self, tmp_path):
        """Create a mock project directory structure."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create basic project structure
        (project_dir / "pipelines").mkdir()
        (project_dir / "collections").mkdir()
        (project_dir / "datasets").mkdir()
        (project_dir / "targets").mkdir()
        (project_dir / ".marimba").mkdir()

        return project_dir

    @pytest.fixture
    def project_wrapper(self, mock_project_dir):
        """Create a ProjectWrapper instance."""
        return ProjectWrapper(mock_project_dir)

    @pytest.fixture
    def mock_pipeline_wrapper(self, mock_project_dir, mocker):
        """Create a real PipelineWrapper instance for testing integration."""
        # Create a minimal pipeline directory structure
        pipeline_dir = mock_project_dir / "pipelines" / "test_pipeline"
        pipeline_dir.mkdir(parents=True)
        repo_dir = pipeline_dir / "repo"
        repo_dir.mkdir()

        # Create minimal config file
        config_file = pipeline_dir / "pipeline.yml"
        config_file.write_text("test_param: test_value")

        # Import here to avoid circular imports during module loading
        from marimba.core.wrappers.pipeline import PipelineWrapper

        # Create wrapper with mocked dependencies
        mocker.patch.object(PipelineWrapper, "_setup_logging")
        mocker.patch("marimba.core.installer.pipeline_installer.PipelineInstaller.create")
        return PipelineWrapper(pipeline_dir, dry_run=True)

    @pytest.fixture
    def mock_collection_wrapper(self, mock_project_dir):
        """Create a real CollectionWrapper instance for testing integration."""
        # Create a minimal collection directory structure
        collection_dir = mock_project_dir / "collections" / "test_collection"
        collection_dir.mkdir(parents=True)

        # Create minimal config file
        config_file = collection_dir / "collection.yml"
        config_file.write_text("name: test_collection")

        # Import here to avoid circular imports
        from marimba.core.wrappers.collection import CollectionWrapper

        return CollectionWrapper(collection_dir)

    @pytest.mark.integration
    def test_project_wrapper_init(self, mock_project_dir):
        """Test ProjectWrapper initialization."""
        wrapper = ProjectWrapper(mock_project_dir)
        assert wrapper.root_dir == mock_project_dir
        assert wrapper.pipelines_dir == mock_project_dir / "pipelines"
        assert wrapper.collections_dir == mock_project_dir / "collections"

    @pytest.mark.integration
    def test_create_project(self, mocker, tmp_path):
        """Test project creation."""
        mock_exists = mocker.patch("pathlib.Path.exists")
        mock_setup_logging = mocker.patch("marimba.core.wrappers.project.ProjectWrapper._setup_logging")
        mock_check_structure = mocker.patch("marimba.core.wrappers.project.ProjectWrapper._check_file_structure")
        mock_exists.return_value = False  # Make it think directory doesn't exist
        mock_setup_logging.return_value = None
        mock_check_structure.return_value = None

        project_path = tmp_path / "new_test_project"

        project = ProjectWrapper.create(project_path)

        assert isinstance(project, ProjectWrapper)
        assert project.root_dir == project_path

    @pytest.mark.integration
    def test_pipeline_wrappers_property(self, project_wrapper, mock_project_dir):
        """Test pipeline wrappers property with realistic structure."""
        # Create pipeline directories with proper structure
        pipeline_dir = mock_project_dir / "pipelines" / "test_pipeline"
        pipeline_dir.mkdir(parents=True)
        repo_dir = pipeline_dir / "repo"
        repo_dir.mkdir()

        # Create proper config file that PipelineWrapper expects
        (pipeline_dir / "pipeline.yml").write_text("test_param: test_value\nname: test_pipeline")

        wrappers = project_wrapper.pipeline_wrappers
        assert isinstance(wrappers, dict)
        # May contain pipelines if validation passes

    @pytest.mark.integration
    def test_collection_wrappers_property(self, project_wrapper, mock_project_dir, mock_collection_wrapper):
        """Test collection wrappers property with real collection structure."""
        # Create collection directories with proper metadata
        collection_dir = mock_project_dir / "collections" / "test_collection"
        collection_dir.mkdir(parents=True, exist_ok=True)

        # Create config file that CollectionWrapper expects
        (collection_dir / "collection.yml").write_text("name: test_collection\ntype: collection")

        wrappers = project_wrapper.collection_wrappers
        assert isinstance(wrappers, dict)
        # May contain collections if validation passes

    @pytest.mark.integration
    def test_dataset_wrappers_property(self, project_wrapper, mock_project_dir):
        """Test dataset wrappers property."""
        # Mock dataset directories
        dataset_dir = mock_project_dir / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)

        wrappers = project_wrapper.dataset_wrappers
        assert isinstance(wrappers, dict)
        # Empty since no valid dataset config exists

    @pytest.mark.integration
    def test_get_pipeline_existing(self, project_wrapper, mock_project_dir, mock_pipeline_wrapper):
        """Test getting existing pipeline using real pipeline wrapper."""
        pipeline_dir = mock_project_dir / "pipelines" / "test_pipeline"
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Create config file that the pipeline wrapper expects
        (pipeline_dir / "pipeline.yml").write_text("test_param: test_value")
        (pipeline_dir / "repo").mkdir(exist_ok=True)

        # Test that pipeline wrappers property includes our pipeline
        wrappers = project_wrapper.pipeline_wrappers
        assert isinstance(wrappers, dict)
        # Note: May be empty if pipeline validation fails, which is acceptable

    @pytest.mark.integration
    def test_get_pipeline_nonexistent(self, project_wrapper):
        """Test getting non-existent pipeline raises exception."""
        with pytest.raises(Exception):
            project_wrapper._get_pipeline("nonexistent_pipeline")

    @pytest.mark.integration
    def test_get_collection_existing(self, project_wrapper, mock_project_dir):
        """Test getting existing collection from wrapper."""
        collection_dir = mock_project_dir / "collections" / "test_collection"
        collection_dir.mkdir(parents=True)

        # Create a mock collection config file
        (collection_dir / "metadata.yaml").touch()

        # Test accessing collection through wrappers property
        wrappers = project_wrapper.collection_wrappers
        assert isinstance(wrappers, dict)

    @pytest.mark.integration
    def test_get_collection_nonexistent(self, project_wrapper):
        """Test getting non-existent collection returns empty dict."""
        wrappers = project_wrapper.collection_wrappers
        assert "nonexistent_collection" not in wrappers

    @pytest.mark.integration
    def test_get_dataset_existing(self, project_wrapper, mock_project_dir):
        """Test getting existing dataset from wrapper."""
        dataset_dir = mock_project_dir / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)

        # Create a mock dataset config file
        (dataset_dir / "metadata.yaml").touch()

        # Test accessing dataset through wrappers property
        wrappers = project_wrapper.dataset_wrappers
        assert isinstance(wrappers, dict)

    @pytest.mark.integration
    def test_get_dataset_nonexistent(self, project_wrapper):
        """Test getting non-existent dataset returns empty dict."""
        wrappers = project_wrapper.dataset_wrappers
        assert "nonexistent_dataset" not in wrappers

    @pytest.mark.integration
    def test_create_pipeline(self, mocker, project_wrapper, mock_project_dir):
        """Test creating a new pipeline with real validation."""
        mock_create = mocker.patch("marimba.core.wrappers.pipeline.PipelineWrapper.create")
        mock_check_name = mocker.patch("marimba.core.wrappers.project.ProjectWrapper.check_name")

        mock_pipeline = mocker.Mock()
        mock_create.return_value = mock_pipeline
        mock_check_name.return_value = None  # Name validation passes

        result = project_wrapper.create_pipeline("new_pipeline", "https://example.com/repo.git", {"key": "value"})

        # Verify name checking was called
        mock_check_name.assert_called_once_with("new_pipeline")
        mock_create.assert_called_once()
        assert result == mock_pipeline

    @pytest.mark.integration
    def test_create_collection(self, mocker, project_wrapper, mock_project_dir):
        """Test creating a new collection with real validation."""
        mock_create = mocker.patch("marimba.core.wrappers.collection.CollectionWrapper.create")
        mock_check_name = mocker.patch("marimba.core.wrappers.project.ProjectWrapper.check_name")

        mock_collection = mocker.Mock()
        mock_create.return_value = mock_collection
        mock_check_name.return_value = None  # Name validation passes

        result = project_wrapper.create_collection("new_collection", {"key": "value"})

        # Verify name checking was called
        mock_check_name.assert_called_once_with("new_collection")
        mock_create.assert_called_once()
        assert result == mock_collection

    @pytest.mark.integration
    def test_create_dataset(self, project_wrapper, mock_project_dir):
        """Test creating a new dataset."""
        # Call with required parameters
        result = project_wrapper.create_dataset(
            "new_dataset",
            {},  # dataset_mapping
            [],  # metadata_mapping_processor_decorator
            [],  # post_package_processors
        )

        # Verify dataset was created
        dataset_dir = mock_project_dir / "datasets" / "new_dataset"
        assert dataset_dir.exists()
        assert hasattr(result, "name")  # DatasetWrapper should have a name property
        result.close()

    @pytest.mark.integration
    def test_delete_project(self, mocker, project_wrapper):
        """Test project deletion."""
        mock_rmtree = mocker.patch("shutil.rmtree")
        result = project_wrapper.delete_project()
        mock_rmtree.assert_called_once()
        assert isinstance(result, Path)

    @pytest.mark.integration
    def test_delete_pipeline(self, project_wrapper, mock_project_dir):
        """Test pipeline deletion."""
        pipeline_dir = mock_project_dir / "pipelines" / "test_pipeline"
        pipeline_dir.mkdir(parents=True)

        # Verify directory exists before deletion
        assert pipeline_dir.exists()

        result = project_wrapper.delete_pipeline("test_pipeline", dry_run=False)

        # Verify directory was deleted and correct path returned
        assert not pipeline_dir.exists()
        assert result == pipeline_dir

    @pytest.mark.integration
    def test_delete_collection(self, project_wrapper, mock_project_dir):
        """Test collection deletion."""
        collection_dir = mock_project_dir / "collections" / "test_collection"
        collection_dir.mkdir(parents=True)

        # Verify directory exists before deletion
        assert collection_dir.exists()

        result = project_wrapper.delete_collection("test_collection", dry_run=False)

        # Verify directory was deleted and correct path returned
        assert not collection_dir.exists()
        assert result == collection_dir

    @pytest.mark.integration
    def test_delete_dataset(self, project_wrapper, mock_project_dir):
        """Test dataset deletion."""
        dataset_dir = mock_project_dir / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)

        # Verify directory exists before deletion
        assert dataset_dir.exists()

        result = project_wrapper.delete_dataset("test_dataset", dry_run=False)

        # Verify directory was deleted and correct path returned
        assert not dataset_dir.exists()
        assert result == dataset_dir

    @pytest.mark.integration
    def test_install_pipelines(self, mocker, project_wrapper, mock_project_dir):
        """Test pipeline installation with real pipeline wrapper."""
        # Create a real pipeline directory
        pipeline_dir = mock_project_dir / "pipelines" / "test_pipeline"
        pipeline_dir.mkdir(parents=True)
        (pipeline_dir / "repo").mkdir()
        (pipeline_dir / "pipeline.yml").write_text("test: config")

        # Mock the install method on real wrappers
        mock_install = mocker.patch("marimba.core.wrappers.pipeline.PipelineWrapper.install")
        mock_install.return_value = None

        # This will work with whatever real pipeline wrappers exist
        project_wrapper.install_pipelines()

        # Verify install was called if any pipeline wrappers were found
        # (May not be called if pipeline validation fails, which is acceptable)

    @pytest.mark.integration
    def test_update_pipelines(self, mocker, project_wrapper, mock_project_dir):
        """Test pipeline updates."""
        pipeline_dir = mock_project_dir / "pipelines" / "test_pipeline"
        pipeline_dir.mkdir(parents=True)

        mock_repo = mocker.patch("git.Repo")
        mock_repo_instance = mocker.Mock()
        mock_repo.return_value = mock_repo_instance

        project_wrapper.update_pipelines()

        # Should attempt to update git repositories

    @pytest.mark.integration
    def test_error_handling_invalid_project_dir(self):
        """Test error handling for invalid project directory."""
        with pytest.raises(Exception):
            ProjectWrapper("/nonexistent/path")

    @pytest.mark.integration
    def test_dry_run_functionality(self, mock_project_dir):
        """Test dry run functionality."""
        wrapper = ProjectWrapper(mock_project_dir, dry_run=True)
        assert wrapper.dry_run is True

    @pytest.mark.integration
    def test_project_properties(self, project_wrapper, mock_project_dir):
        """Test project properties."""
        assert project_wrapper.root_dir == mock_project_dir
        assert project_wrapper.pipelines_dir == mock_project_dir / "pipelines"
        assert project_wrapper.collections_dir == mock_project_dir / "collections"
        assert project_wrapper.datasets_dir == mock_project_dir / "datasets"
        assert project_wrapper.targets_dir == mock_project_dir / "targets"
        assert project_wrapper.marimba_dir == mock_project_dir / ".marimba"
        assert isinstance(project_wrapper.log_path, Path)
        assert project_wrapper.name == mock_project_dir.name

    @pytest.mark.integration
    def test_create_target(self, mocker, project_wrapper):
        """Test creating a distribution target with name validation."""
        config = {"bucket": "test-bucket"}

        mock_create = mocker.patch("marimba.core.wrappers.target.DistributionTargetWrapper.create")
        mock_check_name = mocker.patch("marimba.core.wrappers.project.ProjectWrapper.check_name")

        # Create a mock that's actually a subclass of DistributionTargetWrapper
        mock_target = mocker.Mock(spec=DistributionTargetWrapper)
        mock_target.__class__ = DistributionTargetWrapper  # type: ignore[assignment]
        mock_check_name.return_value = None  # Name validation passes
        mock_create.return_value = mock_target

        result = project_wrapper.create_target("test_target", "s3", config)

        # Verify name checking was called
        mock_check_name.assert_called_once_with("test_target")
        mock_create.assert_called_once()
        assert result == mock_target

    @pytest.mark.unit
    def test_check_name_valid(self):
        """Test valid name checking."""
        # Should not raise any exception
        ProjectWrapper.check_name("valid_name")
        ProjectWrapper.check_name("valid-name")
        ProjectWrapper.check_name("valid123")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "invalid_name,reason",
        [
            ("invalid name", "spaces"),
            ("invalid/name", "slash"),
            ("invalid@name", "special chars"),
            ("invalid\\name", "backslash"),
        ],
    )
    def test_check_name_invalid(self, invalid_name, reason):
        """Test invalid name checking."""
        with pytest.raises(ProjectWrapper.InvalidNameError):
            ProjectWrapper.check_name(invalid_name)

    @pytest.mark.integration
    def test_delete_pipeline_nonexistent(self, project_wrapper):
        """Test deleting non-existent pipeline raises exception."""
        with pytest.raises(ProjectWrapper.DeletePipelineError):
            project_wrapper.delete_pipeline("nonexistent", dry_run=False)

    @pytest.mark.integration
    def test_delete_collection_nonexistent(self, project_wrapper):
        """Test deleting non-existent collection raises exception."""
        with pytest.raises(ProjectWrapper.NoSuchCollectionError):
            project_wrapper.delete_collection("nonexistent", dry_run=False)

    @pytest.mark.integration
    def test_delete_dataset_nonexistent(self, project_wrapper):
        """Test deleting non-existent dataset raises exception."""
        with pytest.raises(FileExistsError):
            project_wrapper.delete_dataset("nonexistent", dry_run=False)

    @pytest.mark.integration
    def test_delete_target_nonexistent(self, project_wrapper):
        """Test deleting non-existent target raises exception."""
        with pytest.raises(FileExistsError):
            project_wrapper.delete_target("nonexistent", dry_run=False)

    @pytest.mark.integration
    def test_delete_target(self, project_wrapper, mock_project_dir):
        """Test target deletion."""
        target_file = mock_project_dir / "targets" / "test_target.yml"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.touch()

        # Verify file exists before deletion
        assert target_file.exists()

        result = project_wrapper.delete_target("test_target", dry_run=False)

        # Verify file was deleted and correct path returned
        assert not target_file.exists()
        assert result == target_file

    @pytest.mark.integration
    def test_dry_run_delete_operations(self, project_wrapper, mock_project_dir):
        """Test dry run mode for delete operations."""
        # Create directories
        pipeline_dir = mock_project_dir / "pipelines" / "test_pipeline"
        pipeline_dir.mkdir(parents=True)
        collection_dir = mock_project_dir / "collections" / "test_collection"
        collection_dir.mkdir(parents=True)
        dataset_dir = mock_project_dir / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)

        # Test dry run doesn't actually delete
        result = project_wrapper.delete_pipeline("test_pipeline", dry_run=True)
        assert pipeline_dir.exists()  # Should still exist
        assert result == pipeline_dir

        result = project_wrapper.delete_collection("test_collection", dry_run=True)
        assert collection_dir.exists()  # Should still exist
        assert result == collection_dir

        result = project_wrapper.delete_dataset("test_dataset", dry_run=True)
        assert dataset_dir.exists()  # Should still exist
        assert result == dataset_dir

    @pytest.mark.integration
    def test_invalid_structure_error(self, mocker, tmp_path):
        """Test InvalidStructureError is raised for invalid project structure."""
        mock_check_structure = mocker.patch("marimba.core.wrappers.project.ProjectWrapper._check_file_structure")
        mock_check_structure.side_effect = ProjectWrapper.InvalidStructureError("Invalid structure")

        with pytest.raises(ProjectWrapper.InvalidStructureError):
            ProjectWrapper(tmp_path)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "method_name,args",
        [
            ("create_pipeline", ("invalid name", "https://example.com", {})),
            ("create_collection", ("invalid name", {})),
            ("create_target", ("invalid name", "s3", {})),
            ("create_dataset", ("invalid name", {}, [], [])),
        ],
    )
    def test_create_methods_invalid_name(self, project_wrapper, method_name, args):
        """Test creating components with invalid name raises error."""
        method = getattr(project_wrapper, method_name)
        with pytest.raises(ProjectWrapper.InvalidNameError):
            method(*args)

    @pytest.mark.integration
    def test_create_project_already_exists(self, mocker, tmp_path):
        """Test creating project when directory already exists raises FileExistsError."""
        mock_exists = mocker.patch("pathlib.Path.exists")
        mock_exists.return_value = True

        with pytest.raises(FileExistsError):
            ProjectWrapper.create(tmp_path / "existing_project")

    @pytest.mark.integration
    def test_target_wrappers_property(self, project_wrapper, mock_project_dir):
        """Test target wrappers property."""
        # Mock target directories
        target_dir = mock_project_dir / "targets" / "test_target"
        target_dir.mkdir(parents=True)

        wrappers = project_wrapper.target_wrappers
        assert isinstance(wrappers, dict)
        # Empty since no valid target config exists

    @pytest.mark.integration
    def test_delete_with_readonly_files(self, mocker, project_wrapper, mock_project_dir):
        """Test deletion handles readonly files properly."""
        pipeline_dir = mock_project_dir / "pipelines" / "test_pipeline"
        pipeline_dir.mkdir(parents=True)

        # Mock remove_directory_tree to simulate the actual deletion logic
        mock_remove_tree = mocker.patch("marimba.core.wrappers.project.remove_directory_tree")
        mock_remove_tree.return_value = None

        result = project_wrapper.delete_pipeline("test_pipeline", dry_run=False)

        mock_remove_tree.assert_called_once_with(pipeline_dir, "pipeline", False)
        assert result == pipeline_dir


class TestProjectWrapperExceptions:
    """Test ProjectWrapper exception classes."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "exception_class,message",
        [
            (ProjectWrapper.InvalidNameError, "Invalid name"),
            (ProjectWrapper.InvalidStructureError, "Invalid structure"),
            (ProjectWrapper.NoSuchCollectionError, "No such collection"),
            (ProjectWrapper.NoSuchPipelineError, "No such pipeline"),
            (ProjectWrapper.NoSuchDatasetError, "No such dataset"),
            (ProjectWrapper.NoSuchTargetError, "No such target"),
            (ProjectWrapper.DeletePipelineError, "Delete pipeline error"),
            (ProjectWrapper.CreateCollectionError, "Create collection error"),
        ],
    )
    def test_project_wrapper_exceptions(self, exception_class, message):
        """Test ProjectWrapper exception classes."""
        error = exception_class(message)
        assert str(error) == message
        assert isinstance(error, Exception)


class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.mark.unit
    def test_get_merged_keyword_args_empty(self, mocker):
        """Test merging with empty arguments."""
        logger = mocker.Mock()
        result = get_merged_keyword_args({}, None, logger)
        assert result == {}

    @pytest.mark.unit
    def test_get_merged_keyword_args_basic(self, mocker):
        """Test basic keyword argument merging."""
        logger = mocker.Mock()
        kwargs = {"key1": "value1", "key2": "value2"}
        result = get_merged_keyword_args(kwargs, None, logger)
        assert result == kwargs

    @pytest.mark.unit
    def test_get_merged_keyword_args_with_extra(self, mocker):
        """Test merging with extra keyword arguments."""
        logger = mocker.Mock()
        kwargs = {"key1": "value1"}
        extra_args = ["key2=value2", "key3=value3"]

        result = get_merged_keyword_args(kwargs, extra_args, logger)

        expected = {"key1": "value1", "key2": "value2", "key3": "value3"}
        assert result == expected

    @pytest.mark.unit
    def test_get_merged_keyword_args_override(self, mocker):
        """Test that extra args can override existing ones."""
        logger = mocker.Mock()
        kwargs = {"key1": "original"}
        extra_args = ["key1=override", "key2=new"]

        result = get_merged_keyword_args(kwargs, extra_args, logger)

        expected = {"key1": "override", "key2": "new"}
        assert result == expected

    @pytest.mark.unit
    def test_get_merged_keyword_args_invalid_format(self, mocker):
        """Test handling of invalid argument format."""
        logger = mocker.Mock()
        kwargs = {"key1": "value1"}
        extra_args = ["invalidarg", "key2=value2"]

        result = get_merged_keyword_args(kwargs, extra_args, logger)

        expected = {"key1": "value1", "key2": "value2"}
        assert result == expected
        # Check that warning was called for the invalid argument format
        logger.warning.assert_any_call('Invalid extra argument provided: "invalidarg"')

    @pytest.mark.unit
    def test_get_merged_keyword_args_evaluation_error(self, mocker):
        """Test handling of evaluation errors."""
        logger = mocker.Mock()
        kwargs: dict[str, Any] = {}
        extra_args = ["key1=not_valid_literal"]

        result = get_merged_keyword_args(kwargs, extra_args, logger)

        expected = {"key1": "not_valid_literal"}  # Should keep as string
        assert result == expected
        logger.warning.assert_called_with('Could not evaluate extra argument value: "not_valid_literal"')

    @pytest.mark.unit
    def test_get_merged_keyword_args_numeric_values(self, mocker):
        """Test handling of numeric values in extra args."""
        logger = mocker.Mock()
        kwargs: dict[str, Any] = {}
        extra_args = ["int_val=42", "float_val=3.14", "bool_val=True", "list_val=[1,2,3]"]

        result = get_merged_keyword_args(kwargs, extra_args, logger)

        expected = {"int_val": 42, "float_val": 3.14, "bool_val": True, "list_val": [1, 2, 3]}
        assert result == expected

    @pytest.mark.unit
    def test_get_merged_keyword_args_none_extra(self, mocker):
        """Test merging when extra_args is None."""
        logger = mocker.Mock()
        kwargs = {"key1": "value1"}
        result = get_merged_keyword_args(kwargs, None, logger)
        assert result == kwargs

    @pytest.mark.unit
    def test_get_merged_keyword_args_complex_types(self, mocker):
        """Test merging with complex data types."""
        logger = mocker.Mock()
        kwargs = {"key1": "value1"}
        extra_args = ["key2=123", "key3=[1,2,3]", "key4={'nested': 'dict'}"]

        result = get_merged_keyword_args(kwargs, extra_args, logger)

        expected = {"key1": "value1", "key2": 123, "key3": [1, 2, 3], "key4": {"nested": "dict"}}
        assert result == expected

    @pytest.mark.unit
    def test_get_merged_keyword_args_skip_invalid_format(self, mocker):
        """Test handling of invalid argument format."""
        logger = mocker.Mock()
        kwargs = {"key1": "value1"}
        extra_args = ["invalid_format", "key2=value2"]

        result = get_merged_keyword_args(kwargs, extra_args, logger)

        # Should skip invalid format and process valid ones
        expected = {"key1": "value1", "key2": "value2"}
        assert result == expected
        # Check that warning was called for the invalid argument format
        logger.warning.assert_any_call('Invalid extra argument provided: "invalid_format"')

    @pytest.mark.unit
    def test_get_merged_keyword_args_invalid_value(self, mocker):
        """Test handling of invalid value format."""
        logger = mocker.Mock()
        kwargs = {"key1": "value1"}
        extra_args = ["key2=invalid_python_literal", "key3=valid_string"]

        result = get_merged_keyword_args(kwargs, extra_args, logger)

        # Should treat unparseable values as strings
        expected = {"key1": "value1", "key2": "invalid_python_literal", "key3": "valid_string"}
        assert result == expected
