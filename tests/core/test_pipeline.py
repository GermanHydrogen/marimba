"""
Test cases for the BasePipeline abstract base class.

This module contains comprehensive tests for the BasePipeline class functionality,
including initialization, abstract methods, command execution, and logging.
"""

import tempfile
from pathlib import Path
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
        self,
        data_dir: Path,
        config: dict[str, Any],
        **kwargs: Any,
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


class TestBasePipelineInitialization:
    """Test cases for BasePipeline initialization."""

    @pytest.mark.unit
    def test_init_with_string_path(self):
        """Test initialization with string root path."""
        root_path = "/test/path"
        config = {"key": "value"}

        pipeline = ConcretePipeline(root_path, config, dry_run=True)

        assert pipeline._root_path == root_path
        assert pipeline._config == config
        assert pipeline._metadata_class == BaseMetadata
        assert pipeline._dry_run == True

    @pytest.mark.unit
    def test_init_with_path_object(self):
        """Test initialization with Path object."""
        root_path = Path("/test/path")

        pipeline = ConcretePipeline(root_path)

        assert pipeline._root_path == root_path
        assert pipeline._config is None
        assert pipeline._metadata_class == BaseMetadata
        assert pipeline._dry_run == False

    @pytest.mark.unit
    def test_init_with_custom_metadata_class(self):
        """Test initialization with custom metadata class."""

        class CustomMetadata(BaseMetadata):
            pass

        pipeline = ConcretePipeline("/test", metadata_class=CustomMetadata)  # type: ignore

        assert pipeline._metadata_class == CustomMetadata

    @pytest.mark.unit
    def test_default_values(self):
        """Test that default values are set correctly."""
        pipeline = ConcretePipeline("/test")

        assert pipeline.config is None
        assert pipeline.dry_run == False
        assert pipeline._metadata_class == BaseMetadata


class TestBasePipelineProperties:
    """Test cases for BasePipeline properties."""

    @pytest.mark.unit
    def test_config_property(self):
        """Test the config property."""
        config = {"test": "value", "number": 42}
        pipeline = ConcretePipeline("/test", config=config)

        assert pipeline.config == config

    @pytest.mark.unit
    def test_config_property_none(self):
        """Test the config property when None."""
        pipeline = ConcretePipeline("/test")

        assert pipeline.config is None

    @pytest.mark.unit
    def test_dry_run_property(self):
        """Test the dry_run property."""
        pipeline = ConcretePipeline("/test", dry_run=True)

        assert pipeline.dry_run == True

    @pytest.mark.unit
    def test_dry_run_property_false(self):
        """Test the dry_run property when False."""
        pipeline = ConcretePipeline("/test")

        assert pipeline.dry_run == False

    @pytest.mark.unit
    def test_class_name_property(self):
        """Test the class_name property."""
        pipeline = ConcretePipeline("/test")

        assert pipeline.class_name == "ConcretePipeline"


class TestBasePipelineStaticMethods:
    """Test cases for BasePipeline static methods."""

    @pytest.mark.unit
    def test_get_pipeline_config_schema_default(self):
        """Test the default pipeline config schema."""
        schema = BasePipeline.get_pipeline_config_schema()

        assert schema == {}
        assert isinstance(schema, dict)

    @pytest.mark.unit
    def test_get_collection_config_schema_default(self):
        """Test the default collection config schema."""
        schema = BasePipeline.get_collection_config_schema()

        assert schema == {}
        assert isinstance(schema, dict)

    @pytest.mark.unit
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

        assert pipeline_schema == {"custom_key": "default_value"}
        assert collection_schema == {"collection_key": 123}


class TestBasePipelineAbstractMethods:
    """Test cases for abstract method enforcement."""

    @pytest.mark.unit
    def test_abstract_pipeline_cannot_be_instantiated(self):
        """Test that BasePipeline cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BasePipeline("/test")  # type: ignore

        assert "abstract" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_incomplete_implementation_cannot_be_instantiated(self):
        """Test that incomplete implementations cannot be instantiated."""
        with pytest.raises(TypeError) as exc_info:
            AbstractOnlyPipeline("/test")  # type: ignore

        assert "abstract" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_concrete_implementation_can_be_instantiated(self):
        """Test that complete implementations can be instantiated."""
        # This should not raise an exception
        pipeline = ConcretePipeline("/test")
        assert isinstance(pipeline, BasePipeline)


class TestRunImportCommand:
    """Test cases for the run_import command."""

    @pytest.mark.integration
    def test_run_import_success(self, tmp_path, mocker):
        """Test successful import command execution."""
        root_path = tmp_path / "project" / "pipelines" / "test_pipeline"
        data_dir = tmp_path / "data"
        source_dir = tmp_path / "source"

        root_path.mkdir(parents=True)
        data_dir.mkdir()
        source_dir.mkdir()

        pipeline = ConcretePipeline(str(root_path))

        mock_format_path = mocker.patch("marimba.core.pipeline.format_path_for_logging")
        mock_format_path.return_value = "formatted/path"
        config = {"test": "config"}
        kwargs: dict[str, Any] = {"extra": "args"}

        mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
        mock_logger = mocker.Mock()
        mock_get_logger.return_value = mock_logger
        pipeline.run_import(data_dir, source_dir, config, **kwargs)

        # Verify the _import method was called with correct arguments
        assert pipeline.import_called
        assert pipeline.last_import_args == (data_dir, source_dir, config, kwargs)

        # Verify logging calls
        assert mock_logger.info.call_count == 2
        expected_message = (
            "Started [steel_blue3]import[/steel_blue3] command for pipeline "
            f"[light_pink3]ConcretePipeline[/light_pink3] with args data_dir=formatted/path, "
            f"source_path={source_dir}, config={{'test': 'config'}}, kwargs={{'extra': 'args'}}"
        )
        mock_logger.info.assert_any_call(expected_message)
        mock_logger.info.assert_any_call(
            "Completed [steel_blue3]import[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3]",
        )

    @pytest.mark.integration
    def test_run_import_invalid_source_path(self, mocker):
        """Test import command with invalid source path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir) / "project" / "pipelines" / "test_pipeline"
            data_dir = Path(temp_dir) / "data"
            invalid_source = Path(temp_dir) / "nonexistent"

            root_path.mkdir(parents=True)
            data_dir.mkdir()

            pipeline = ConcretePipeline(str(root_path))
            config: dict[str, Any] = {}

            mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
            mock_logger = mocker.Mock()
            mock_get_logger.return_value = mock_logger
            pipeline.run_import(data_dir, invalid_source, config)

            # Verify _import was not called
            assert not pipeline.import_called

            # Verify error was logged
            mock_logger.exception.assert_called_once_with(f"Source path {invalid_source} is not a directory")

    @pytest.mark.integration
    def test_run_import_source_path_is_file(self, mocker):
        """Test import command when source path is a file, not directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir) / "project" / "pipelines" / "test_pipeline"
            data_dir = Path(temp_dir) / "data"
            source_file = Path(temp_dir) / "source.txt"

            root_path.mkdir(parents=True)
            data_dir.mkdir()
            source_file.touch()

            pipeline = ConcretePipeline(str(root_path))
            config: dict[str, Any] = {}

            mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
            mock_logger = mocker.Mock()
            mock_get_logger.return_value = mock_logger
            pipeline.run_import(data_dir, source_file, config)

            # Verify _import was not called
            assert not pipeline.import_called

            # Verify error was logged
            mock_logger.exception.assert_called_once_with(f"Source path {source_file} is not a directory")


class TestRunProcessCommand:
    """Test cases for the run_process command."""

    @pytest.mark.integration
    def test_run_process_success(self, mocker):
        """Test successful process command execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir) / "project" / "pipelines" / "test_pipeline"
            data_dir = Path(temp_dir) / "data"

            root_path.mkdir(parents=True)
            data_dir.mkdir()

            pipeline = ConcretePipeline(str(root_path))

            mock_format_path = mocker.patch("marimba.core.pipeline.format_path_for_logging")
            mock_format_path.return_value = "formatted/path"
            config = {"process": "config"}
            kwargs: dict[str, Any] = {"additional": "parameters"}

            mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
            mock_logger = mocker.Mock()
            mock_get_logger.return_value = mock_logger
            pipeline.run_process(data_dir, config, **kwargs)

            # Verify the _process method was called with correct arguments
            assert pipeline.process_called
            assert pipeline.last_process_args == (data_dir, config, kwargs)

            # Verify logging calls
            assert mock_logger.info.call_count == 2
            mock_logger.info.assert_any_call(
                "Started [steel_blue3]process[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3] with args "
                "data_dir=formatted/path, config={'process': 'config'}, kwargs={'additional': 'parameters'}",
            )
            mock_logger.info.assert_any_call(
                "Completed [steel_blue3]process[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3]",
            )


class TestRunPackageCommand:
    """Test cases for the run_package command."""

    @pytest.mark.integration
    def test_run_package_success(self, mocker):
        """Test successful package command execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir) / "project" / "pipelines" / "test_pipeline"
            data_dir = Path(temp_dir) / "data"

            root_path.mkdir(parents=True)
            data_dir.mkdir()

            pipeline = ConcretePipeline(str(root_path))

            mock_format_path = mocker.patch("marimba.core.pipeline.format_path_for_logging")
            mock_format_path.return_value = "formatted/path"
            config = {"package": "config"}
            kwargs: dict[str, Any] = {"extra": "options"}

            mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
            mock_logger = mocker.Mock()
            mock_get_logger.return_value = mock_logger
            result = pipeline.run_package(data_dir, config, **kwargs)

            # Verify the _package method was called with correct arguments
            assert pipeline.package_called
            assert pipeline.last_package_args == (data_dir, config, kwargs)

            # Verify return value
            expected_result: dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]] = {
                Path("source1.txt"): (Path("dest1.txt"), [], {"meta": "data1"}),
                Path("source2.txt"): (Path("dest2.txt"), None, None),
            }
            assert result == expected_result

            # Verify logging calls
            assert mock_logger.info.call_count == 2
            mock_logger.info.assert_any_call(
                "Started [steel_blue3]package[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3] with args "
                "data_dir=formatted/path, config={'package': 'config'}, kwargs={'extra': 'options'}",
            )
            mock_logger.info.assert_any_call(
                "Completed [steel_blue3]package[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3]",
            )


class TestRunPostPackageCommand:
    """Test cases for the run_post_package command."""

    @pytest.mark.integration
    def test_run_post_package_success(self, mocker):
        """Test successful post package command execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir) / "project" / "pipelines" / "test_pipeline"
            dataset_dir = Path(temp_dir) / "dataset"

            root_path.mkdir(parents=True)
            dataset_dir.mkdir()

            pipeline = ConcretePipeline(str(root_path))

            mock_format_path = mocker.patch("marimba.core.pipeline.format_path_for_logging")
            mock_format_path.return_value = "formatted/path"

            mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
            mock_logger = mocker.Mock()
            mock_get_logger.return_value = mock_logger
            result = pipeline.run_post_package(dataset_dir)

            # Verify the _post_package method was called with correct arguments
            assert pipeline.post_package_called
            assert pipeline.last_post_package_args == (dataset_dir,)

            # Verify return value
            expected_result = {Path("changed1.txt"), Path("changed2.txt")}
            assert result == expected_result

            # Verify logging calls
            assert mock_logger.info.call_count == 2
            mock_logger.info.assert_any_call(
                "Started [steel_blue3]post package[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3] with args "
                "dataset_dir=formatted/path",
            )
            mock_logger.info.assert_any_call(
                "Completed [steel_blue3]post package[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3]",
            )


class TestDefaultImplementations:
    """Test cases for default implementations of optional methods."""

    @pytest.mark.unit
    def test_import_default_warning(self, mocker):
        """Test that default _import implementation logs a warning."""

        # Create a pipeline that uses the default _import method
        class DefaultImportPipeline(BasePipeline):
            def _package(
                self,
                data_dir: Path,
                config: dict[str, Any],
                **kwargs: Any,
            ) -> dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]]:
                return {}

        pipeline = DefaultImportPipeline("/test")
        mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
        mock_logger = mocker.Mock()
        mock_get_logger.return_value = mock_logger
        # Call the default implementation
        pipeline._import(Path("/data"), Path("/source"), {})

        mock_logger.warning.assert_called_once_with(
            "There is no Marimba [steel_blue3]import[/steel_blue3] command implemented for pipeline [light_pink3]DefaultImportPipeline[/light_pink3]",
        )

    @pytest.mark.unit
    def test_process_default_warning(self, mocker):
        """Test that default _process implementation logs a warning."""

        # Create a pipeline that uses the default _process method
        class DefaultProcessPipeline(BasePipeline):
            def _package(
                self,
                data_dir: Path,
                config: dict[str, Any],
                **kwargs: Any,
            ) -> dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]]:
                return {}

        pipeline = DefaultProcessPipeline("/test")
        mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
        mock_logger = mocker.Mock()
        mock_get_logger.return_value = mock_logger
        # Call the default implementation
        pipeline._process(Path("/data"), {})

        mock_logger.warning.assert_called_once_with(
            "There is no Marimba [steel_blue3]process[/steel_blue3] command implemented for pipeline [light_pink3]DefaultProcessPipeline[/light_pink3]",
        )

    @pytest.mark.unit
    def test_post_package_default_returns_empty_set(self):
        """Test that default _post_package implementation returns empty set."""
        # Call the default implementation directly
        pipeline = ConcretePipeline("/test")
        result = BasePipeline._post_package(pipeline, Path("/dataset"))

        assert result == set()
        assert isinstance(result, set)


class TestErrorHandlingAndEdgeCases:
    """Test cases for error handling and edge cases."""

    @pytest.mark.unit
    def test_pipeline_with_empty_config(self):
        """Test pipeline behavior with empty configuration."""
        pipeline = ConcretePipeline("/test", config={})

        assert pipeline.config == {}
        assert pipeline.config is not None

    @pytest.mark.unit
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

        assert pipeline.config == config

    @pytest.mark.unit
    def test_logging_with_none_kwargs(self, mocker):
        """Test that logging works correctly with None or empty kwargs."""
        mock_format_path = mocker.patch("marimba.core.pipeline.format_path_for_logging")
        mock_format_path.return_value = "formatted/path"
        pipeline = ConcretePipeline("/test/project/pipelines/test")

        mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
        mock_logger = mocker.Mock()
        mock_get_logger.return_value = mock_logger
        pipeline.run_process(Path("/data"), {})

        # Should not raise an exception and should log correctly
        mock_logger.info.assert_any_call(
            "Started [steel_blue3]process[/steel_blue3] command for pipeline [light_pink3]ConcretePipeline[/light_pink3] with args "
            "data_dir=formatted/path, config={}, kwargs={}",
        )

    @pytest.mark.integration
    def test_path_formatting_in_logs(self, mocker):
        """Test that path formatting is called correctly in logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir) / "project" / "pipelines" / "test"
            root_path.mkdir(parents=True)
            data_path = Path(temp_dir) / "data"
            data_path.mkdir()

            pipeline = ConcretePipeline(str(root_path))

            mock_format = mocker.patch("marimba.core.pipeline.format_path_for_logging")
            mock_format.return_value = "mocked/path"
            mock_get_logger = mocker.patch("marimba.core.utils.log.get_logger")
            mock_logger = mocker.Mock()
            mock_get_logger.return_value = mock_logger
            pipeline.run_process(data_path, {})

            # Verify format_path_for_logging was called with correct arguments
            expected_parent = Path(temp_dir)  # root_path.parents[2]
            mock_format.assert_called_with(data_path, expected_parent)


if __name__ == "__main__":
    pytest.main([__file__])
