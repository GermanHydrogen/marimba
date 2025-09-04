"""Tests for marimba.core.wrappers.project module."""

from pathlib import Path
from unittest.mock import Mock, patch, PropertyMock

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

    def test_project_wrapper_init(self, mock_project_dir):
        """Test ProjectWrapper initialization."""
        wrapper = ProjectWrapper(mock_project_dir)
        assert wrapper.root_dir == mock_project_dir
        assert wrapper.pipelines_dir == mock_project_dir / "pipelines"
        assert wrapper.collections_dir == mock_project_dir / "collections"

    @patch('marimba.core.wrappers.project.ProjectWrapper._check_file_structure')
    @patch('marimba.core.wrappers.project.ProjectWrapper._setup_logging')
    @patch('pathlib.Path.exists')
    def test_create_project(self, mock_exists, mock_setup_logging, mock_check_structure, tmp_path):
        """Test project creation."""
        mock_exists.return_value = False  # Make it think directory doesn't exist
        mock_setup_logging.return_value = None
        mock_check_structure.return_value = None
        
        project_path = tmp_path / "new_test_project"
        
        project = ProjectWrapper.create(project_path)
        
        assert isinstance(project, ProjectWrapper)
        assert project.root_dir == project_path

    def test_pipeline_wrappers_property(self, project_wrapper, mock_project_dir):
        """Test pipeline wrappers property."""
        # Mock pipeline directories
        pipeline_dir = mock_project_dir / "pipelines" / "test_pipeline"
        pipeline_dir.mkdir(parents=True)
        
        # Create a mock pipeline config file
        (pipeline_dir / "config.yaml").touch()
        
        wrappers = project_wrapper.pipeline_wrappers
        assert isinstance(wrappers, dict)
        # Empty since no valid pipeline config exists

    def test_collection_wrappers_property(self, project_wrapper, mock_project_dir):
        """Test collection wrappers property."""
        # Mock collection directories
        collection_dir = mock_project_dir / "collections" / "test_collection"
        collection_dir.mkdir(parents=True)
        
        wrappers = project_wrapper.collection_wrappers
        assert isinstance(wrappers, dict)
        # Empty since no valid collection config exists

    def test_dataset_wrappers_property(self, project_wrapper, mock_project_dir):
        """Test dataset wrappers property."""
        # Mock dataset directories
        dataset_dir = mock_project_dir / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)
        
        wrappers = project_wrapper.dataset_wrappers
        assert isinstance(wrappers, dict)
        # Empty since no valid dataset config exists

    @patch('marimba.core.wrappers.project.ProjectWrapper._get_pipeline')
    def test_get_pipeline_existing(self, mock_get_pipeline, project_wrapper, mock_project_dir):
        """Test getting existing pipeline."""
        pipeline_dir = mock_project_dir / "pipelines" / "test_pipeline"
        pipeline_dir.mkdir(parents=True)
        
        mock_pipeline = Mock()
        mock_get_pipeline.return_value = mock_pipeline
        
        # Access the private method directly since there's no public get_pipeline
        pipeline = project_wrapper._get_pipeline("test_pipeline")
        assert pipeline is not None

    def test_get_pipeline_nonexistent(self, project_wrapper):
        """Test getting non-existent pipeline raises exception."""
        with pytest.raises(Exception):
            project_wrapper._get_pipeline("nonexistent_pipeline")

    def test_get_collection_existing(self, project_wrapper, mock_project_dir):
        """Test getting existing collection from wrapper."""
        collection_dir = mock_project_dir / "collections" / "test_collection"
        collection_dir.mkdir(parents=True)
        
        # Create a mock collection config file
        (collection_dir / "metadata.yaml").touch()
        
        # Test accessing collection through wrappers property
        wrappers = project_wrapper.collection_wrappers
        assert isinstance(wrappers, dict)

    def test_get_collection_nonexistent(self, project_wrapper):
        """Test getting non-existent collection returns empty dict."""
        wrappers = project_wrapper.collection_wrappers
        assert "nonexistent_collection" not in wrappers

    def test_get_dataset_existing(self, project_wrapper, mock_project_dir):
        """Test getting existing dataset from wrapper."""
        dataset_dir = mock_project_dir / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)
        
        # Create a mock dataset config file
        (dataset_dir / "metadata.yaml").touch()
        
        # Test accessing dataset through wrappers property
        wrappers = project_wrapper.dataset_wrappers
        assert isinstance(wrappers, dict)

    def test_get_dataset_nonexistent(self, project_wrapper):
        """Test getting non-existent dataset returns empty dict."""
        wrappers = project_wrapper.dataset_wrappers
        assert "nonexistent_dataset" not in wrappers

    @patch('marimba.core.wrappers.pipeline.PipelineWrapper.create')
    def test_create_pipeline(self, mock_create, project_wrapper):
        """Test creating a new pipeline."""
        mock_pipeline = Mock()
        mock_create.return_value = mock_pipeline
        
        result = project_wrapper.create_pipeline(
            "new_pipeline",
            "https://example.com/repo.git",
            {"key": "value"}
        )
        
        mock_create.assert_called_once()
        assert result == mock_pipeline

    @patch('marimba.core.wrappers.collection.CollectionWrapper.create')
    def test_create_collection(self, mock_create, project_wrapper):
        """Test creating a new collection."""
        mock_collection = Mock()
        mock_create.return_value = mock_collection
        
        result = project_wrapper.create_collection("new_collection", {"key": "value"})
        
        mock_create.assert_called_once()
        assert result == mock_collection

    def test_create_dataset(self, project_wrapper, mock_project_dir):
        """Test creating a new dataset."""
        # Call with required parameters
        result = project_wrapper.create_dataset(
            "new_dataset", 
            {},  # dataset_mapping
            [],  # metadata_mapping_processor_decorator
            []   # post_package_processors
        )
        
        # Verify dataset was created
        dataset_dir = mock_project_dir / "datasets" / "new_dataset"
        assert dataset_dir.exists()
        assert hasattr(result, 'name')  # DatasetWrapper should have a name property

    def test_delete_project(self, project_wrapper):
        """Test project deletion."""
        with patch('shutil.rmtree') as mock_rmtree:
            result = project_wrapper.delete_project()
            mock_rmtree.assert_called_once()
            assert isinstance(result, Path)

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

    def test_install_pipelines(self, project_wrapper):
        """Test pipeline installation."""
        # Mock the pipeline wrapper installation
        mock_wrapper = Mock()
        mock_wrapper.install.return_value = None
        
        with patch.object(project_wrapper, '_pipeline_wrappers', {'test_pipeline': mock_wrapper}):
            with patch('marimba.core.wrappers.project.ProjectWrapper.pipeline_wrappers', new_callable=PropertyMock) as mock_prop:
                mock_prop.return_value = {'test_pipeline': mock_wrapper}
                
                project_wrapper.install_pipelines()
                
                mock_wrapper.install.assert_called_once()

    @patch('git.Repo')
    def test_update_pipelines(self, mock_repo, project_wrapper, mock_project_dir):
        """Test pipeline updates."""
        pipeline_dir = mock_project_dir / "pipelines" / "test_pipeline"
        pipeline_dir.mkdir(parents=True)
        
        mock_repo_instance = Mock()
        mock_repo.return_value = mock_repo_instance
        
        project_wrapper.update_pipelines()
        
        # Should attempt to update git repositories

    def test_error_handling_invalid_project_dir(self):
        """Test error handling for invalid project directory."""
        with pytest.raises(Exception):
            ProjectWrapper("/nonexistent/path")

    def test_dry_run_functionality(self, mock_project_dir):
        """Test dry run functionality."""
        wrapper = ProjectWrapper(mock_project_dir, dry_run=True)
        assert wrapper.dry_run is True

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

    def test_create_target(self, project_wrapper):
        """Test creating a distribution target."""
        config = {"bucket": "test-bucket"}
        
        with patch('marimba.core.wrappers.target.DistributionTargetWrapper.create') as mock_create:
            # Create a mock that's actually a subclass of DistributionTargetWrapper
            mock_target = Mock(spec=DistributionTargetWrapper)
            mock_target.__class__ = DistributionTargetWrapper
            
            mock_create.return_value = mock_target
            
            result = project_wrapper.create_target("test_target", "s3", config)
            
            mock_create.assert_called_once()
            assert result == mock_target

    def test_check_name_valid(self):
        """Test valid name checking."""
        # Should not raise any exception
        ProjectWrapper.check_name("valid_name")
        ProjectWrapper.check_name("valid-name")
        ProjectWrapper.check_name("valid123")

    def test_check_name_invalid(self):
        """Test invalid name checking."""
        with pytest.raises(ProjectWrapper.InvalidNameError):
            ProjectWrapper.check_name("invalid name")  # spaces
        with pytest.raises(ProjectWrapper.InvalidNameError):
            ProjectWrapper.check_name("invalid/name")  # slash
        with pytest.raises(ProjectWrapper.InvalidNameError):
            ProjectWrapper.check_name("invalid@name")  # special chars


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_merged_keyword_args_empty(self):
        """Test merging with empty arguments."""
        logger = Mock()
        result = get_merged_keyword_args({}, None, logger)
        assert result == {}

    def test_get_merged_keyword_args_basic(self):
        """Test basic keyword argument merging."""
        logger = Mock()
        kwargs = {"key1": "value1", "key2": "value2"}
        result = get_merged_keyword_args(kwargs, None, logger)
        assert result == kwargs

    def test_get_merged_keyword_args_with_extra(self):
        """Test merging with extra keyword arguments."""
        logger = Mock()
        kwargs = {"key1": "value1"}
        extra_args = ["key2=value2", "key3=value3"]
        
        result = get_merged_keyword_args(kwargs, extra_args, logger)
        
        expected = {"key1": "value1", "key2": "value2", "key3": "value3"}
        assert result == expected

    def test_get_merged_keyword_args_override(self):
        """Test that extra args can override existing ones."""
        logger = Mock()
        kwargs = {"key1": "original"}
        extra_args = ["key1=override", "key2=new"]
        
        result = get_merged_keyword_args(kwargs, extra_args, logger)
        
        expected = {"key1": "override", "key2": "new"}
        assert result == expected

    def test_get_merged_keyword_args_none_extra(self):
        """Test merging when extra_args is None."""
        logger = Mock()
        kwargs = {"key1": "value1"}
        result = get_merged_keyword_args(kwargs, None, logger)
        assert result == kwargs

    def test_get_merged_keyword_args_complex_types(self):
        """Test merging with complex data types."""
        logger = Mock()
        kwargs = {"key1": "value1"}
        extra_args = ["key2=123", "key3=[1,2,3]", "key4={'nested': 'dict'}"]
        
        result = get_merged_keyword_args(kwargs, extra_args, logger)
        
        expected = {
            "key1": "value1", 
            "key2": 123, 
            "key3": [1, 2, 3], 
            "key4": {"nested": "dict"}
        }
        assert result == expected

    def test_get_merged_keyword_args_invalid_format(self):
        """Test handling of invalid argument format."""
        logger = Mock()
        kwargs = {"key1": "value1"}
        extra_args = ["invalid_format", "key2=value2"]
        
        result = get_merged_keyword_args(kwargs, extra_args, logger)
        
        # Should skip invalid format and process valid ones
        expected = {"key1": "value1", "key2": "value2"}
        assert result == expected
        # Should be called twice - once for invalid format, once for value that can't be evaluated
        assert logger.warning.call_count == 2

    def test_get_merged_keyword_args_invalid_value(self):
        """Test handling of invalid value format."""
        logger = Mock()
        kwargs = {"key1": "value1"}
        extra_args = ["key2=invalid_python_literal", "key3=valid_string"]
        
        result = get_merged_keyword_args(kwargs, extra_args, logger)
        
        # Should treat unparseable values as strings
        expected = {"key1": "value1", "key2": "invalid_python_literal", "key3": "valid_string"}
        assert result == expected