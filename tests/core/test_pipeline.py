"""
Test cases for the BasePipeline abstract base class.

This module contains comprehensive tests for the BasePipeline class functionality,
including initialization, abstract methods, command execution, and logging.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from unittest import TestCase
from typing import Any

import pytest

from marimba.core.pipeline import BasePipeline
from marimba.core.schemas.base import BaseMetadata


class ConcretePipeline(BasePipeline):
    """Concrete implementation of BasePipeline for testing purposes."""

    def __init__(
        self,
        root_path: Path | str,
        config: dict[str, Any] | None = None,
        metadata_class: type[BaseMetadata] = BaseMetadata,
        *,
        dry_run: bool = False,
    ) -> None:
        super().__init__(root_path, config, metadata_class, dry_run=dry_run)
        # Track method calls for testing
        self.import_called = False
        self.process_called = False
        self.package_called = False
        self.post_package_called = False

    def _import(self, data_dir: Path, source_path: Path, config: dict[str, Any], **kwargs: Any) -> None:
        self.import_called = True
        self.last_import_args = (data_dir, source_path, config, kwargs)

    def _process(self, data_dir: Path, config: dict[str, Any], **kwargs: Any) -> None:
        self.process_called = True
        self.last_process_args = (data_dir, config, kwargs)

    def _package(
        self, data_dir: Path, config: dict[str, Any], **kwargs: Any
    ) -> dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]]:
        self.package_called = True
        self.last_package_args = (data_dir, config, kwargs)
        return {
            Path("source1.txt"): (Path("dest1.txt"), [], {"meta": "data1"}),
            Path("source2.txt"): (Path("dest2.txt"), None, None),
        }

    def _post_package(self, dataset_dir: Path) -> set[Path]:
        self.post_package_called = True
        self.last_post_package_args = (dataset_dir,)
        return {Path("changed1.txt"), Path("changed2.txt")}


class AbstractOnlyPipeline(BasePipeline):
    """Pipeline that doesn't implement abstract methods for testing."""

    pass


class TestBasePipelineInitialization(TestCase):
    """Test cases for BasePipeline initialization."""

    def test_init_with_string_path(self):
        """Test initialization with string root path."""
        root_path = "/test/path"
        config = {"key": "value"}

        pipeline = ConcretePipeline(root_path, config, dry_run=True)

        self.assertEqual(pipeline._root_path, root_path)
        self.assertEqual(pipeline._config, config)
        self.assertEqual(pipeline._metadata_class, BaseMetadata)
        self.assertTrue(pipeline._dry_run)

    def test_init_with_path_object(self):
        """Test initialization with Path object."""
        root_path = Path("/test/path")

        pipeline = ConcretePipeline(root_path)

        self.assertEqual(pipeline._root_path, root_path)
        self.assertIsNone(pipeline._config)
        self.assertEqual(pipeline._metadata_class, BaseMetadata)
        self.assertFalse(pipeline._dry_run)

    def test_init_with_custom_metadata_class(self):
        """Test initialization with custom metadata class."""

        class CustomMetadata(BaseMetadata):
            pass

        pipeline = ConcretePipeline("/test", metadata_class=CustomMetadata)  # type: ignore

        self.assertEqual(pipeline._metadata_class, CustomMetadata)

    def test_default_values(self):
        """Test that default values are set correctly."""
        pipeline = ConcretePipeline("/test")

        self.assertIsNone(pipeline.config)
        self.assertFalse(pipeline.dry_run)
        self.assertEqual(pipeline._metadata_class, BaseMetadata)


class TestBasePipelineProperties(TestCase):
    """Test cases for BasePipeline properties."""

    def test_config_property(self):
        """Test the config property."""
        config = {"test": "value", "number": 42}
        pipeline = ConcretePipeline("/test", config=config)

        self.assertEqual(pipeline.config, config)

    def test_config_property_none(self):
        """Test the config property when None."""
        pipeline = ConcretePipeline("/test")

        self.assertIsNone(pipeline.config)

    def test_dry_run_property(self):
        """Test the dry_run property."""
        pipeline = ConcretePipeline("/test", dry_run=True)

        self.assertTrue(pipeline.dry_run)

    def test_dry_run_property_false(self):
        """Test the dry_run property when False."""
        pipeline = ConcretePipeline("/test")

        self.assertFalse(pipeline.dry_run)

    def test_class_name_property(self):
        """Test the class_name property."""
        pipeline = ConcretePipeline("/test")

        self.assertEqual(pipeline.class_name, "ConcretePipeline")


class TestBasePipelineStaticMethods(TestCase):
    """Test cases for BasePipeline static methods."""

    def test_get_pipeline_config_schema_default(self):
        """Test the default pipeline config schema."""
        schema = BasePipeline.get_pipeline_config_schema()

        self.assertEqual(schema, {})
        self.assertIsInstance(schema, dict)

    def test_get_collection_config_schema_default(self):
        """Test the default collection config schema."""
        schema = BasePipeline.get_collection_config_schema()

        self.assertEqual(schema, {})
        self.assertIsInstance(schema, dict)

    def test_static_methods_can_be_overridden(self):
        """Test that static methods can be overridden in subclasses."""

        class CustomPipeline(ConcretePipeline):
            @staticmethod
            def get_pipeline_config_schema() -> dict[str, Any]:
                return {"custom_key": "default_value"}

            @staticmethod
            def get_collection_config_schema() -> dict[str, Any]:
                return {"collection_key": 123}

        pipeline_schema = CustomPipeline.get_pipeline_config_schema()
        collection_schema = CustomPipeline.get_collection_config_schema()

        self.assertEqual(pipeline_schema, {"custom_key": "default_value"})
        self.assertEqual(collection_schema, {"collection_key": 123})


class TestBasePipelineAbstractMethods(TestCase):
    """Test cases for abstract method enforcement."""

    def test_abstract_pipeline_cannot_be_instantiated(self):
        """Test that BasePipeline cannot be instantiated directly."""
        with self.assertRaises(TypeError) as context:
            BasePipeline("/test")  # type: ignore

        self.assertIn("abstract", str(context.exception).lower())

    def test_incomplete_implementation_cannot_be_instantiated(self):
        """Test that incomplete implementations cannot be instantiated."""
        with self.assertRaises(TypeError) as context:
            AbstractOnlyPipeline("/test")  # type: ignore

        self.assertIn("abstract", str(context.exception).lower())

    def test_concrete_implementation_can_be_instantiated(self):
        """Test that complete implementations can be instantiated."""
        # This should not raise an exception
        pipeline = ConcretePipeline("/test")
        self.assertIsInstance(pipeline, BasePipeline)


class TestRunImportCommand(TestCase):
    """Test cases for the run_import command."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name) / "project" / "pipelines" / "test_pipeline"
        self.data_dir = Path(self.temp_dir.name) / "data"
        self.source_dir = Path(self.temp_dir.name) / "source"

        self.root_path.mkdir(parents=True)
        self.data_dir.mkdir()
        self.source_dir.mkdir()

        self.pipeline = ConcretePipeline(str(self.root_path))

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch("marimba.core.pipeline.format_path_for_logging")
    def test_run_import_success(self, mock_format_path):
        """Test successful import command execution."""
        mock_format_path.return_value = "formatted/path"
        config = {"test": "config"}
        kwargs: dict[str, Any] = {"extra": "args"}

        with patch("marimba.core.utils.log.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            self.pipeline.run_import(self.data_dir, self.source_dir, config, **kwargs)

            # Verify the _import method was called with correct arguments
            self.assertTrue(self.pipeline.import_called)
            self.assertEqual(self.pipeline.last_import_args, (self.data_dir, self.source_dir, config, kwargs))

            # Verify logging calls
            self.assertEqual(mock_logger.info.call_count, 2)
            mock_logger.info.assert_any_call(
                "Started [steel_blue3]import[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3] with args "
                "data_dir=formatted/path, source_path=%s, config={'test': 'config'}, kwargs={'extra': 'args'}"
                % self.source_dir
            )
            mock_logger.info.assert_any_call(
                "Completed [steel_blue3]import[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3]"
            )

    def test_run_import_invalid_source_path(self):
        """Test import command with invalid source path."""
        invalid_source = Path(self.temp_dir.name) / "nonexistent"
        config: dict[str, Any] = {}

        with patch("marimba.core.utils.log.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            self.pipeline.run_import(self.data_dir, invalid_source, config)

            # Verify _import was not called
            self.assertFalse(self.pipeline.import_called)

            # Verify error was logged
            mock_logger.exception.assert_called_once_with(f"Source path {invalid_source} is not a directory")

    def test_run_import_source_path_is_file(self):
        """Test import command when source path is a file, not directory."""
        source_file = Path(self.temp_dir.name) / "source.txt"
        source_file.touch()
        config: dict[str, Any] = {}

        with patch("marimba.core.utils.log.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            self.pipeline.run_import(self.data_dir, source_file, config)

            # Verify _import was not called
            self.assertFalse(self.pipeline.import_called)

            # Verify error was logged
            mock_logger.exception.assert_called_once_with(f"Source path {source_file} is not a directory")


class TestRunProcessCommand(TestCase):
    """Test cases for the run_process command."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name) / "project" / "pipelines" / "test_pipeline"
        self.data_dir = Path(self.temp_dir.name) / "data"

        self.root_path.mkdir(parents=True)
        self.data_dir.mkdir()

        self.pipeline = ConcretePipeline(str(self.root_path))

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch("marimba.core.pipeline.format_path_for_logging")
    def test_run_process_success(self, mock_format_path):
        """Test successful process command execution."""
        mock_format_path.return_value = "formatted/path"
        config = {"process": "config"}
        kwargs: dict[str, Any] = {"additional": "parameters"}

        with patch("marimba.core.utils.log.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            self.pipeline.run_process(self.data_dir, config, **kwargs)

            # Verify the _process method was called with correct arguments
            self.assertTrue(self.pipeline.process_called)
            self.assertEqual(self.pipeline.last_process_args, (self.data_dir, config, kwargs))

            # Verify logging calls
            self.assertEqual(mock_logger.info.call_count, 2)
            mock_logger.info.assert_any_call(
                "Started [steel_blue3]process[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3] with args "
                "data_dir=formatted/path, config={'process': 'config'}, kwargs={'additional': 'parameters'}"
            )
            mock_logger.info.assert_any_call(
                "Completed [steel_blue3]process[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3]"
            )


class TestRunPackageCommand(TestCase):
    """Test cases for the run_package command."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name) / "project" / "pipelines" / "test_pipeline"
        self.data_dir = Path(self.temp_dir.name) / "data"

        self.root_path.mkdir(parents=True)
        self.data_dir.mkdir()

        self.pipeline = ConcretePipeline(str(self.root_path))

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch("marimba.core.pipeline.format_path_for_logging")
    def test_run_package_success(self, mock_format_path):
        """Test successful package command execution."""
        mock_format_path.return_value = "formatted/path"
        config = {"package": "config"}
        kwargs: dict[str, Any] = {"extra": "options"}

        with patch("marimba.core.utils.log.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            result = self.pipeline.run_package(self.data_dir, config, **kwargs)

            # Verify the _package method was called with correct arguments
            self.assertTrue(self.pipeline.package_called)
            self.assertEqual(self.pipeline.last_package_args, (self.data_dir, config, kwargs))

            # Verify return value
            expected_result: dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]] = {
                Path("source1.txt"): (Path("dest1.txt"), [], {"meta": "data1"}),
                Path("source2.txt"): (Path("dest2.txt"), None, None),
            }
            self.assertEqual(result, expected_result)

            # Verify logging calls
            self.assertEqual(mock_logger.info.call_count, 2)
            mock_logger.info.assert_any_call(
                "Started [steel_blue3]package[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3] with args "
                "data_dir=formatted/path, config={'package': 'config'}, kwargs={'extra': 'options'}"
            )
            mock_logger.info.assert_any_call(
                "Completed [steel_blue3]package[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3]"
            )


class TestRunPostPackageCommand(TestCase):
    """Test cases for the run_post_package command."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name) / "project" / "pipelines" / "test_pipeline"
        self.dataset_dir = Path(self.temp_dir.name) / "dataset"

        self.root_path.mkdir(parents=True)
        self.dataset_dir.mkdir()

        self.pipeline = ConcretePipeline(str(self.root_path))

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch("marimba.core.pipeline.format_path_for_logging")
    def test_run_post_package_success(self, mock_format_path):
        """Test successful post package command execution."""
        mock_format_path.return_value = "formatted/path"

        with patch("marimba.core.utils.log.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            result = self.pipeline.run_post_package(self.dataset_dir)

            # Verify the _post_package method was called with correct arguments
            self.assertTrue(self.pipeline.post_package_called)
            self.assertEqual(self.pipeline.last_post_package_args, (self.dataset_dir,))

            # Verify return value
            expected_result = {Path("changed1.txt"), Path("changed2.txt")}
            self.assertEqual(result, expected_result)

            # Verify logging calls
            self.assertEqual(mock_logger.info.call_count, 2)
            mock_logger.info.assert_any_call(
                "Started [steel_blue3]post package[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3] with args "
                "dataset_dir=formatted/path"
            )
            mock_logger.info.assert_any_call(
                "Completed [steel_blue3]post package[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3]"
            )


class TestDefaultImplementations(TestCase):
    """Test cases for default implementations of optional methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = ConcretePipeline("/test")

    def test_import_default_warning(self):
        """Test that default _import implementation logs a warning."""

        # Create a pipeline that uses the default _import method
        class DefaultImportPipeline(BasePipeline):
            def _package(
                self, data_dir: Path, config: dict[str, Any], **kwargs: Any
            ) -> dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]]:
                return {}

        pipeline = DefaultImportPipeline("/test")
        with patch("marimba.core.utils.log.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            # Call the default implementation
            pipeline._import(Path("/data"), Path("/source"), {})

            mock_logger.warning.assert_called_once_with(
                "There is no Marimba [steel_blue3]import[/steel_blue3] command implemented for pipeline [light_pink3]DefaultImportPipeline[/light_pink3]"
            )

    def test_process_default_warning(self):
        """Test that default _process implementation logs a warning."""

        # Create a pipeline that uses the default _process method
        class DefaultProcessPipeline(BasePipeline):
            def _package(
                self, data_dir: Path, config: dict[str, Any], **kwargs: Any
            ) -> dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]]:
                return {}

        pipeline = DefaultProcessPipeline("/test")
        with patch("marimba.core.utils.log.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            # Call the default implementation
            pipeline._process(Path("/data"), {})

            mock_logger.warning.assert_called_once_with(
                "There is no Marimba [steel_blue3]process[/steel_blue3] command implemented for pipeline [light_pink3]DefaultProcessPipeline[/light_pink3]"
            )

    def test_post_package_default_returns_empty_set(self):
        """Test that default _post_package implementation returns empty set."""
        # Call the default implementation directly
        result = BasePipeline._post_package(self.pipeline, Path("/dataset"))

        self.assertEqual(result, set())
        self.assertIsInstance(result, set)


class TestErrorHandlingAndEdgeCases(TestCase):
    """Test cases for error handling and edge cases."""

    def test_pipeline_with_empty_config(self):
        """Test pipeline behavior with empty configuration."""
        pipeline = ConcretePipeline("/test", config={})

        self.assertEqual(pipeline.config, {})
        self.assertIsNotNone(pipeline.config)

    def test_pipeline_with_complex_config(self):
        """Test pipeline with complex configuration data."""
        config = {
            "string_val": "test",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "list_val": [1, 2, 3],
            "dict_val": {"nested": "value"},
        }

        pipeline = ConcretePipeline("/test", config=config)

        self.assertEqual(pipeline.config, config)

    @patch("marimba.core.pipeline.format_path_for_logging")
    def test_logging_with_none_kwargs(self, mock_format_path):
        """Test that logging works correctly with None or empty kwargs."""
        mock_format_path.return_value = "formatted/path"
        pipeline = ConcretePipeline("/test/project/pipelines/test")

        with patch("marimba.core.utils.log.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            pipeline.run_process(Path("/data"), {})

            # Should not raise an exception and should log correctly
            mock_logger.info.assert_any_call(
                "Started [steel_blue3]process[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3] with args "
                "data_dir=formatted/path, config={}, kwargs={}"
            )

    def test_path_formatting_in_logs(self):
        """Test that path formatting is called correctly in logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir) / "project" / "pipelines" / "test"
            root_path.mkdir(parents=True)
            data_path = Path(temp_dir) / "data"
            data_path.mkdir()

            pipeline = ConcretePipeline(str(root_path))

            with patch("marimba.core.pipeline.format_path_for_logging") as mock_format:
                mock_format.return_value = "mocked/path"
                with patch("marimba.core.utils.log.get_logger") as mock_get_logger:
                    mock_logger = Mock()
                    mock_get_logger.return_value = mock_logger
                    pipeline.run_process(data_path, {})

                    # Verify format_path_for_logging was called with correct arguments
                    expected_parent = Path(temp_dir)  # root_path.parents[2]
                    mock_format.assert_called_with(data_path, expected_parent)


if __name__ == "__main__":
    pytest.main([__file__])
