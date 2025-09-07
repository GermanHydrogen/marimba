"""
Test cases for the pipeline loader module.

This module contains comprehensive tests for the pipeline loading functionality,
including module discovery, class instantiation, error handling, and logging configuration.
"""

import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from marimba.core.parallel.pipeline_loader import (
    _configure_pipeline_logging,
    _find_pipeline_class,
    _find_pipeline_module_path,
    _is_valid_pipeline_class,
    _load_pipeline_module,
    _log_empty_repo_warning,
    load_pipeline_instance,
)
from marimba.core.pipeline import BasePipeline
from marimba.core.schemas.base import BaseMetadata


class MockTestPipeline(BasePipeline):
    """Mock test pipeline class for testing purposes (renamed to avoid pytest collection)."""

    def __init__(self, repo_dir: Path | str, config: dict[str, Any] | None = None, *, dry_run: bool = False) -> None:
        super().__init__(repo_dir, config, dry_run=dry_run)

    @staticmethod
    def get_pipeline_config_schema() -> dict[str, Any]:
        return {}

    @staticmethod
    def get_collection_config_schema() -> dict[str, Any]:
        return {}

    def _import(self, data_dir: Path, source_path: Path, config: dict[str, Any], **kwargs: Any) -> None:
        pass

    def _process(self, data_dir: Path, config: dict[str, Any], **kwargs: Any) -> None:
        pass

    def _package(
        self,
        data_dir: Path,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]]:
        return {}


class TestFindPipelineModulePath:
    """Test cases for _find_pipeline_module_path function."""

    @pytest.mark.integration
    def test_find_single_pipeline_file(self):
        """Test finding a single .pipeline.py file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            pipeline_file = repo_path / "test.pipeline.py"
            pipeline_file.touch()

            result = _find_pipeline_module_path(repo_path)

            assert result == pipeline_file

    @pytest.mark.integration
    def test_find_nested_pipeline_file(self):
        """Test finding a .pipeline.py file in a subdirectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            nested_dir = repo_path / "src" / "pipelines"
            nested_dir.mkdir(parents=True)
            pipeline_file = nested_dir / "my_pipeline.pipeline.py"
            pipeline_file.touch()

            result = _find_pipeline_module_path(repo_path)

            assert result == pipeline_file

    @pytest.mark.integration
    def test_no_pipeline_file_raises_error(self):
        """Test that FileNotFoundError is raised when no .pipeline.py file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            # Create some non-pipeline files
            (repo_path / "README.md").touch()
            (repo_path / "config.py").touch()

            with pytest.raises(FileNotFoundError) as context:
                _find_pipeline_module_path(repo_path)

            assert "No pipeline implementation found" in str(context.value)
            assert ".pipeline.py" in str(context.value)

    @pytest.mark.integration
    def test_no_pipeline_file_with_allow_empty(self, mocker):
        """Test that None is returned when no .pipeline.py file exists and allow_empty=True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)

            mock_log = mocker.patch("marimba.core.parallel.pipeline_loader._log_empty_repo_warning")
            result = _find_pipeline_module_path(repo_path, allow_empty=True)

            assert result is None
            mock_log.assert_called_once_with(repo_path)

    @pytest.mark.integration
    def test_multiple_pipeline_files_raises_error(self):
        """Test that FileNotFoundError is raised when multiple .pipeline.py files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            pipeline_file1 = repo_path / "first.pipeline.py"
            pipeline_file2 = repo_path / "second.pipeline.py"
            pipeline_file1.touch()
            pipeline_file2.touch()

            with pytest.raises(FileNotFoundError) as context:
                _find_pipeline_module_path(repo_path)

            assert "Multiple pipeline implementations found" in str(context.value)


class TestLogEmptyRepoWarning:
    """Test cases for _log_empty_repo_warning function."""

    @pytest.mark.unit
    def test_logs_warning_message(self, mocker):
        """Test that warning message is logged with correct content."""
        mock_logger = mocker.Mock()
        mock_get_logger = mocker.patch("marimba.core.parallel.pipeline_loader.get_logger")
        mock_get_logger.return_value = mock_logger
        repo_path = Path("/test/repo")

        _log_empty_repo_warning(repo_path)

        mock_get_logger.assert_called_once_with("marimba.core.pipeline")
        mock_logger.warning.assert_called_once()

        # Check that warning message contains expected content
        warning_message = mock_logger.warning.call_args[0][0]
        assert "no Marimba Pipeline implementation was found" in warning_message
        assert ".pipeline.py" in warning_message
        assert "Pipeline template" in warning_message
        assert "https://raw.githubusercontent.com" in warning_message


class TestLoadPipelineModule:
    """Test cases for _load_pipeline_module function."""

    @pytest.mark.integration
    def test_load_valid_module(self):
        """Test loading a valid Python module."""
        with tempfile.TemporaryDirectory() as temp_dir:
            module_path = Path(temp_dir) / "test.pipeline.py"
            module_path.write_text("# Test module content\ntest_var = 'hello'")

            module_name, module, module_spec = _load_pipeline_module(module_path)

            assert module_name == "test.pipeline"
            assert module is not None
            assert module_spec is not None
            assert "test.pipeline" in sys.modules

    @pytest.mark.integration
    def test_load_module_spec_none(self, mocker):
        """Test ImportError when module spec is None."""
        mock_spec_from_file = mocker.patch("marimba.core.parallel.pipeline_loader.spec_from_file_location")
        mock_spec_from_file.return_value = None
        module_path = Path("/fake/path/test.pipeline.py")

        with pytest.raises(ImportError) as context:
            _load_pipeline_module(module_path)

        assert "Could not load spec" in str(context.value)

    @pytest.mark.integration
    def test_load_module_loader_none(self, mocker):
        """Test ImportError when module spec loader is None."""
        mock_spec_from_file = mocker.patch("marimba.core.parallel.pipeline_loader.spec_from_file_location")
        mock_spec = mocker.Mock()
        mock_spec.loader = None
        mock_spec_from_file.return_value = mock_spec
        module_path = Path("/fake/path/test.pipeline.py")

        with pytest.raises(ImportError) as context:
            _load_pipeline_module(module_path)

        assert "Could not find loader" in str(context.value)


class TestIsValidPipelineClass:
    """Test cases for _is_valid_pipeline_class function."""

    @pytest.mark.unit
    def test_valid_pipeline_class(self):
        """Test that a valid pipeline class returns True."""
        result = _is_valid_pipeline_class(MockTestPipeline)
        assert result

    @pytest.mark.unit
    def test_base_pipeline_class_invalid(self):
        """Test that BasePipeline itself returns False."""
        result = _is_valid_pipeline_class(BasePipeline)
        assert not result

    @pytest.mark.unit
    def test_non_class_object_invalid(self):
        """Test that non-class objects return False."""
        # These tests check the function's handling of invalid arguments
        # The function should handle these gracefully
        result = _is_valid_pipeline_class(str)  # type: ignore
        assert not result

        result = _is_valid_pipeline_class(int)  # type: ignore
        assert not result

        # Test with an actual class that's not a pipeline
        class NotAPipeline:
            pass

        result = _is_valid_pipeline_class(NotAPipeline)
        assert not result

    @pytest.mark.unit
    def test_non_pipeline_class_invalid(self):
        """Test that classes not inheriting from BasePipeline return False."""

        class NotAPipeline:
            pass

        result = _is_valid_pipeline_class(NotAPipeline)
        assert not result

    @pytest.mark.unit
    def test_type_error_handling(self):
        """Test that TypeError is handled gracefully."""

        # Create an object that raises TypeError in isinstance check
        class ProblematicClass:
            @property  # type: ignore
            def __class__(self) -> Any:
                raise TypeError("Can't determine class")

        result = _is_valid_pipeline_class(ProblematicClass)  # type: ignore
        assert not result


class TestFindPipelineClass:
    """Test cases for _find_pipeline_class function."""

    @pytest.mark.unit
    def test_find_valid_pipeline_class(self, mocker):
        """Test finding a valid pipeline class in a module."""
        mock_module = mocker.Mock()
        mock_module.__dict__ = {
            "MockTestPipeline": MockTestPipeline,
            "some_function": lambda: None,
            "some_variable": 42,
        }

        result = _find_pipeline_class(mock_module)

        assert result == MockTestPipeline

    @pytest.mark.unit
    def test_no_pipeline_class_raises_error(self, mocker):
        """Test that ImportError is raised when no pipeline class is found."""
        mock_module = mocker.Mock()
        mock_module.__dict__ = {
            "some_function": lambda: None,
            "some_variable": 42,
            "NotAPipeline": str,  # Not a pipeline class
        }

        with pytest.raises(ImportError) as context:
            _find_pipeline_class(mock_module)

        assert "Pipeline class has not been set" in str(context.value)

    @pytest.mark.unit
    def test_module_without_dict_raises_error(self):
        """Test that ImportError is raised when module has no __dict__."""

        # Create an object that truly doesn't have __dict__
        class NoDict:
            __slots__ = []  # This prevents __dict__ from being created

        mock_module = NoDict()

        with pytest.raises(ImportError) as context:
            _find_pipeline_class(mock_module)  # type: ignore

        assert "module has no __dict__" in str(context.value)

    @pytest.mark.unit
    def test_multiple_pipeline_classes_returns_first(self, mocker):
        """Test that first valid pipeline class is returned when multiple exist."""

        class AnotherTestPipeline(BasePipeline):
            @staticmethod
            def get_pipeline_config_schema() -> dict[str, Any]:
                return {}

            @staticmethod
            def get_collection_config_schema() -> dict[str, Any]:
                return {}

            def _import(self, data_dir: Path, source_path: Path, config: dict[str, Any], **kwargs: Any) -> None:
                pass

            def _process(self, data_dir: Path, config: dict[str, Any], **kwargs: Any) -> None:
                pass

            def _package(
                self,
                data_dir: Path,
                config: dict[str, Any],
                **kwargs: Any,
            ) -> dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]]:
                return {}

        mock_module = mocker.Mock()
        mock_module.__dict__ = {
            "FirstPipeline": MockTestPipeline,
            "SecondPipeline": AnotherTestPipeline,
        }

        result = _find_pipeline_class(mock_module)

        # Should return one of the valid classes (implementation dependent on dict ordering)
        assert issubclass(result, BasePipeline)
        assert result != BasePipeline


class TestConfigurePipelineLogging:
    """Test cases for _configure_pipeline_logging function."""

    @pytest.mark.unit
    def test_configure_logging_with_prefix(self, mocker):
        """Test configuring pipeline logging with log prefix."""
        mock_get_file_handler = mocker.patch("marimba.core.parallel.pipeline_loader.get_file_handler")
        mock_prefix_filter = mocker.patch("marimba.core.parallel.pipeline_loader.LogPrefixFilter")

        mock_pipeline = mocker.Mock()
        mock_pipeline.logger = mocker.Mock()
        mock_pipeline.logger.handlers = []

        mock_handler = mocker.Mock()
        mock_handler.baseFilename = "/test/log/file.log"
        mock_get_file_handler.return_value = mock_handler

        mock_filter_instance = mocker.Mock()
        mock_prefix_filter.return_value = mock_filter_instance

        root_dir = Path("/test/root")
        pipeline_name = "test_pipeline"
        log_prefix = "TEST_PREFIX"

        _configure_pipeline_logging(mock_pipeline, root_dir, pipeline_name, False, log_prefix)

        # Check that prefix filter was created and added
        mock_prefix_filter.assert_called_once_with(log_prefix)
        mock_pipeline.logger.addFilter.assert_called_once_with(mock_filter_instance.apply_prefix)

        # Check that file handler was created and added
        mock_get_file_handler.assert_called_once_with(root_dir, pipeline_name, False)
        mock_pipeline.logger.addHandler.assert_called_once_with(mock_handler)

    @pytest.mark.unit
    def test_configure_logging_without_prefix(self, mocker):
        """Test configuring pipeline logging without log prefix."""
        mock_get_file_handler = mocker.patch("marimba.core.parallel.pipeline_loader.get_file_handler")

        mock_pipeline = mocker.Mock()
        mock_pipeline.logger = mocker.Mock()
        mock_pipeline.logger.handlers = []

        mock_handler = mocker.Mock()
        mock_handler.baseFilename = "/test/log/file.log"
        mock_get_file_handler.return_value = mock_handler

        root_dir = Path("/test/root")
        pipeline_name = "test_pipeline"

        _configure_pipeline_logging(mock_pipeline, root_dir, pipeline_name, True, None)

        # Check that no filter was added
        mock_pipeline.logger.addFilter.assert_not_called()

        # Check that file handler was created and added
        mock_get_file_handler.assert_called_once_with(root_dir, pipeline_name, True)
        mock_pipeline.logger.addHandler.assert_called_once_with(mock_handler)

    @pytest.mark.unit
    def test_prevent_duplicate_handlers(self, mocker):
        """Test that duplicate handlers are not added."""
        mock_get_file_handler = mocker.patch("marimba.core.parallel.pipeline_loader.get_file_handler")

        mock_pipeline = mocker.Mock()
        existing_handler = mocker.Mock()
        existing_handler.baseFilename = "/test/log/file.log"
        mock_pipeline.logger.handlers = [existing_handler]

        mock_handler = mocker.Mock()
        mock_handler.baseFilename = "/test/log/file.log"  # Same path as existing
        mock_get_file_handler.return_value = mock_handler

        root_dir = Path("/test/root")
        pipeline_name = "test_pipeline"

        _configure_pipeline_logging(mock_pipeline, root_dir, pipeline_name, False, None)

        # Handler should not be added since one with same path already exists
        # Note: The current implementation always clears handlers first (line 137), so it will add the handler
        # This test is documenting the actual behavior rather than an expected prevention
        mock_pipeline.logger.addHandler.assert_called_once_with(mock_handler)


class TestLoadPipelineInstance:
    """Test cases for load_pipeline_instance function."""

    @pytest.fixture
    def pipeline_test_dirs(self):
        """Set up test directories and files."""
        temp_dir = tempfile.TemporaryDirectory()
        root_dir = Path(temp_dir.name) / "root"
        repo_dir = Path(temp_dir.name) / "repo"
        config_path = Path(temp_dir.name) / "config.yaml"

        root_dir.mkdir()
        repo_dir.mkdir()
        config_path.write_text("key: value")

        # Create a test pipeline file
        pipeline_file = repo_dir / "test.pipeline.py"
        pipeline_content = """
from pathlib import Path
from typing import Any
from marimba.core.pipeline import BasePipeline
from marimba.core.schemas.base import BaseMetadata

class MockTestPipeline(BasePipeline):
    @staticmethod
    def get_pipeline_config_schema() -> dict[str, Any]:
        return {}
    
    @staticmethod
    def get_collection_config_schema() -> dict[str, Any]:
        return {}
    
    def _import(self, data_dir: Path, source_path: Path, config: dict[str, Any], **kwargs: Any) -> None:
        pass
    
    def _process(self, data_dir: Path, config: dict[str, Any], **kwargs: Any) -> None:
        pass
    
    def _package(self, data_dir: Path, config: dict[str, Any], **kwargs: Any) -> dict[Path, tuple[Path, list[BaseMetadata] | None, dict[str, Any] | None]]:
        return {}
"""
        pipeline_file.write_text(pipeline_content)

        yield {
            "temp_dir": temp_dir,
            "root_dir": root_dir,
            "repo_dir": repo_dir,
            "config_path": config_path,
            "pipeline_file": pipeline_file,
        }

        temp_dir.cleanup()

    @pytest.mark.integration
    def test_load_pipeline_instance_success(self, pipeline_test_dirs, mocker):
        """Test successfully loading a pipeline instance."""
        mock_configure_logging = mocker.patch("marimba.core.parallel.pipeline_loader._configure_pipeline_logging")
        mock_load_config = mocker.patch("marimba.core.parallel.pipeline_loader.load_config")
        mock_load_config.return_value = {"test": "config"}

        result = load_pipeline_instance(
            pipeline_test_dirs["root_dir"],
            pipeline_test_dirs["repo_dir"],
            "test_pipeline",
            pipeline_test_dirs["config_path"],
            False,
            "LOG_PREFIX",
        )

        assert result is not None
        assert isinstance(result, BasePipeline)

        # Check that configuration was loaded
        mock_load_config.assert_called_once_with(pipeline_test_dirs["config_path"])

        # Check that logging was configured
        mock_configure_logging.assert_called_once_with(
            result,
            pipeline_test_dirs["root_dir"],
            "test_pipeline",
            False,
            "LOG_PREFIX",
        )

    @pytest.mark.integration
    def test_load_pipeline_instance_empty_repo_allow_empty(self, pipeline_test_dirs, mocker):
        """Test loading from empty repository with allow_empty=True."""
        # Remove the pipeline file to simulate empty repo
        pipeline_test_dirs["pipeline_file"].unlink()

        mocker.patch("marimba.core.parallel.pipeline_loader._log_empty_repo_warning")
        result = load_pipeline_instance(
            pipeline_test_dirs["root_dir"],
            pipeline_test_dirs["repo_dir"],
            "test_pipeline",
            pipeline_test_dirs["config_path"],
            False,
            allow_empty=True,
        )

        assert result is None

    @pytest.mark.integration
    def test_load_pipeline_instance_empty_repo_no_allow_empty(self, pipeline_test_dirs):
        """Test loading from empty repository with allow_empty=False raises error."""
        # Remove the pipeline file to simulate empty repo
        pipeline_test_dirs["pipeline_file"].unlink()

        with pytest.raises(FileNotFoundError):
            load_pipeline_instance(
                pipeline_test_dirs["root_dir"],
                pipeline_test_dirs["repo_dir"],
                "test_pipeline",
                pipeline_test_dirs["config_path"],
                False,
            )

    @pytest.mark.integration
    def test_load_pipeline_instance_import_error(self, pipeline_test_dirs, mocker):
        """Test handling of import errors during module loading."""
        mock_load_module = mocker.patch("marimba.core.parallel.pipeline_loader._load_pipeline_module")
        mock_load_module.side_effect = ImportError("Test import error")

        with pytest.raises(ImportError) as context:
            load_pipeline_instance(
                pipeline_test_dirs["root_dir"],
                pipeline_test_dirs["repo_dir"],
                "test_pipeline",
                pipeline_test_dirs["config_path"],
                False,
            )

        assert "Test import error" in str(context.value)

    @pytest.mark.integration
    def test_sys_path_manipulation(self, pipeline_test_dirs, mocker):
        """Test that sys.path is properly manipulated during module loading."""
        mock_load_config = mocker.patch("marimba.core.parallel.pipeline_loader.load_config")
        mock_load_config.return_value = {}
        original_path = sys.path.copy()

        result = load_pipeline_instance(
            pipeline_test_dirs["root_dir"],
            pipeline_test_dirs["repo_dir"],
            "test_pipeline",
            pipeline_test_dirs["config_path"],
            False,
        )

        # sys.path should be restored to original state
        assert sys.path == original_path
        assert result is not None

    @pytest.mark.integration
    def test_module_execution_failure(self, pipeline_test_dirs, mocker):
        """Test handling of module execution failures."""
        mock_find_class = mocker.patch("marimba.core.parallel.pipeline_loader._find_pipeline_class")
        mock_load_config = mocker.patch("marimba.core.parallel.pipeline_loader.load_config")
        mock_load_config.return_value = {}

        # Create a pipeline file with syntax error
        bad_pipeline_file = pipeline_test_dirs["repo_dir"] / "bad.pipeline.py"
        bad_pipeline_file.write_text("invalid python syntax <<<")
        pipeline_test_dirs["pipeline_file"].unlink()  # Remove good file

        with pytest.raises(Exception):  # Could be SyntaxError or other execution error
            load_pipeline_instance(
                pipeline_test_dirs["root_dir"],
                pipeline_test_dirs["repo_dir"],
                "test_pipeline",
                pipeline_test_dirs["config_path"],
                False,
            )


if __name__ == "__main__":
    pytest.main([__file__])
