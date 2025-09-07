"""Tests for marimba.main CLI module."""

import pytest
import typer
from typer.testing import CliRunner

from marimba.core.utils.log import LogLevel
from marimba.main import (
    global_options,
    marimba_cli,
    version_callback,
)
from tests.conftest import assert_cli_failure, assert_cli_success


class TestCLI:
    """Test CLI functionality."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_project_dir(self, tmp_path):
        """Create a mock project directory."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        return project_dir

    @pytest.mark.unit
    def test_marimba_cli_creation(self):
        """Test that the CLI app is created correctly."""
        assert isinstance(marimba_cli, typer.Typer)
        assert marimba_cli.info.name == "Marimba"
        assert marimba_cli.info.help is not None
        assert "FAIR scientific image datasets" in marimba_cli.info.help

    @pytest.mark.unit
    def test_version_callback_with_version(self, mocker):
        """Test version callback when version is available."""
        mocker.patch("importlib.metadata.version", return_value="1.0.0")
        with pytest.raises(typer.Exit):
            version_callback(True)

    @pytest.mark.unit
    def test_version_callback_without_flag(self):
        """Test version callback when flag is False."""
        # This should not raise an exception or exit
        version_callback(False)

    @pytest.mark.unit
    def test_version_callback_with_exception(self, mocker):
        """Test version callback when metadata is not available."""
        mocker.patch("importlib.metadata.version", side_effect=Exception("No metadata"))
        with pytest.raises(typer.Exit):
            version_callback(True)

    @pytest.mark.integration
    def test_version_command_with_version(self, mocker, runner):
        """Test version command when version is available."""
        mocker.patch("importlib.metadata.version", return_value="1.0.0")
        result = runner.invoke(marimba_cli, ["version"])
        assert_cli_success(result, expected_message="Marimba v1.0.0", context="Version command")

    @pytest.mark.integration
    def test_version_command_with_exception(self, mocker, runner):
        """Test version command when metadata is not available."""
        mocker.patch("importlib.metadata.version", side_effect=Exception("No metadata"))
        result = runner.invoke(marimba_cli, ["version"])
        assert_cli_success(
            result,
            expected_message="unknown (not installed as package)",
            context="Version command without metadata",
        )

    @pytest.mark.unit
    def test_global_options_with_debug(self, mocker):
        """Test global options with debug level."""
        mock_ctx = mocker.Mock()
        mock_ctx.invoked_subcommand = "import"  # Simulate subcommand

        mock_handler = mocker.patch("marimba.main.get_rich_handler")
        mock_log_handler = mocker.Mock()
        mock_handler.return_value = mock_log_handler

        global_options(ctx=mock_ctx, level=LogLevel.DEBUG, version=False)

        # Should set debug level on the handler
        mock_log_handler.setLevel.assert_called_once_with(10)

    @pytest.mark.unit
    def test_global_options_with_quiet(self, mocker):
        """Test global options with error level (quiet)."""
        mock_ctx = mocker.Mock()
        mock_ctx.invoked_subcommand = "import"  # Simulate subcommand

        mock_handler = mocker.patch("marimba.main.get_rich_handler")
        mock_log_handler = mocker.Mock()
        mock_handler.return_value = mock_log_handler

        global_options(ctx=mock_ctx, level=LogLevel.ERROR, version=False)

        # Should set error level on the handler
        mock_log_handler.setLevel.assert_called_once_with(40)

    @pytest.mark.unit
    def test_global_options_normal(self, mocker):
        """Test global options with default settings."""
        mock_ctx = mocker.Mock()
        mock_ctx.invoked_subcommand = "import"  # Simulate subcommand

        mock_handler = mocker.patch("marimba.main.get_rich_handler")
        mock_log_handler = mocker.Mock()
        mock_handler.return_value = mock_log_handler

        global_options(ctx=mock_ctx, level=LogLevel.WARNING, version=False)

        # Should set warning level on the handler (default)
        mock_log_handler.setLevel.assert_called_once_with(30)

    @pytest.mark.integration
    def test_import_command_basic(self, mocker, runner, mock_project_dir):
        """Test basic import command functionality."""
        mocker.patch("marimba.main.validate_dependencies")
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.run_import.return_value = None
        mock_project.collection_wrappers = {}  # Empty collections dict
        mock_project.pipeline_wrappers = {"pipeline1": mocker.Mock()}  # Mock pipelines
        mock_project.prompt_collection_config.return_value = {"test": "config"}
        mock_project.create_collection.return_value = None
        mock_find_project.return_value = mock_project_dir

        # Mock the source path
        source_path = mock_project_dir / "source"
        source_path.mkdir()

        result = runner.invoke(
            marimba_cli,
            ["import", "test_collection", str(source_path), "--project-dir", str(mock_project_dir)],
        )

        assert_cli_success(result, context="Basic import command")
        mock_find_project.assert_called_once()
        mock_project_wrapper.assert_called_once()
        mock_project.run_import.assert_called_once()

    @pytest.mark.integration
    def test_import_command_with_options(self, mocker, runner, mock_project_dir):
        """Test import command with various options."""
        mocker.patch("marimba.main.validate_dependencies")
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.run_import.return_value = None
        mock_project.collection_wrappers = {}
        mock_project.pipeline_wrappers = {"pipeline1": mocker.Mock()}
        mock_project.prompt_collection_config.return_value = {"test": "config"}
        mock_project.create_collection.return_value = None
        mock_find_project.return_value = mock_project_dir

        source_path = mock_project_dir / "source"
        source_path.mkdir()

        result = runner.invoke(
            marimba_cli,
            [
                "import",
                "test_collection",
                str(source_path),
                "--project-dir",
                str(mock_project_dir),
                "--overwrite",
                "--dry-run",
            ],
        )

        assert_cli_success(result, context="Import command with options")
        mock_project.run_import.assert_called_once()

    @pytest.mark.integration
    def test_process_command_basic(self, mocker, runner, mock_project_dir):
        """Test basic process command functionality."""
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.run_process.return_value = None
        mock_project.collection_wrappers = {"test_collection": mocker.Mock()}
        mock_project.pipeline_wrappers = {"test_pipeline": mocker.Mock()}
        mock_find_project.return_value = mock_project_dir

        result = runner.invoke(
            marimba_cli,
            [
                "process",
                "--collection-name",
                "test_collection",
                "--pipeline-name",
                "test_pipeline",
                "--project-dir",
                str(mock_project_dir),
            ],
        )

        assert_cli_success(result, context="Basic process command")
        mock_project_wrapper.assert_called_once()
        mock_project.run_process.assert_called_once()

    @pytest.mark.integration
    def test_process_command_with_options(self, mocker, runner, mock_project_dir):
        """Test process command with various options."""
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.run_process.return_value = None
        mock_project.collection_wrappers = {"test_collection": mocker.Mock()}
        mock_project.pipeline_wrappers = {"test_pipeline": mocker.Mock()}
        mock_find_project.return_value = mock_project_dir

        result = runner.invoke(
            marimba_cli,
            [
                "process",
                "--collection-name",
                "test_collection",
                "--pipeline-name",
                "test_pipeline",
                "--project-dir",
                str(mock_project_dir),
                "--dry-run",
            ],
        )

        assert_cli_success(result, context="Process command with options")
        mock_project.run_process.assert_called_once()

    @pytest.mark.integration
    def test_package_command_basic(self, mocker, runner, mock_project_dir):
        """Test basic package command functionality."""
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.compose.return_value = {"collection1": mocker.Mock()}
        mock_project.create_dataset.return_value = mocker.Mock()
        mock_project.collection_wrappers = {"test_collection": mocker.Mock()}
        mock_project.pipeline_wrappers = {"pipeline1": mocker.Mock()}
        mock_project.get_pipeline_post_processors.return_value = []
        mock_find_project.return_value = mock_project_dir

        result = runner.invoke(
            marimba_cli,
            ["package", "test_dataset", "--collection-name", "test_collection", "--project-dir", str(mock_project_dir)],
        )

        assert_cli_success(result, context="Basic package command")
        mock_project_wrapper.assert_called_once()
        mock_project.compose.assert_called_once()
        mock_project.create_dataset.assert_called_once()

    @pytest.mark.integration
    def test_package_command_with_options(self, mocker, runner, mock_project_dir):
        """Test package command with various options."""
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.compose.return_value = {"collection1": mocker.Mock()}
        mock_project.create_dataset.return_value = mocker.Mock()
        mock_project.collection_wrappers = {"test_collection": mocker.Mock()}
        mock_project.pipeline_wrappers = {"pipeline1": mocker.Mock()}
        mock_project.get_pipeline_post_processors.return_value = []
        mock_find_project.return_value = mock_project_dir

        result = runner.invoke(
            marimba_cli,
            [
                "package",
                "test_dataset",
                "--collection-name",
                "test_collection",
                "--project-dir",
                str(mock_project_dir),
                "--version",
                "2.0",
            ],
        )

        assert_cli_success(result, context="Package command with options")
        mock_project.compose.assert_called_once()
        mock_project.create_dataset.assert_called_once()

    @pytest.mark.integration
    def test_distribute_command_basic(self, mocker, runner, mock_project_dir):
        """Test basic distribute command functionality."""
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.distribute.return_value = None
        mock_find_project.return_value = mock_project_dir

        result = runner.invoke(
            marimba_cli,
            ["distribute", "test_dataset", "test_target", "--project-dir", str(mock_project_dir)],
        )

        assert_cli_success(result, context="Basic distribute command")
        mock_project_wrapper.assert_called_once()
        mock_project.distribute.assert_called_once()

    @pytest.mark.integration
    def test_update_command_basic(self, mocker, runner, mock_project_dir):
        """Test basic update command functionality."""
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.update_pipelines.return_value = None
        mock_find_project.return_value = mock_project_dir

        result = runner.invoke(marimba_cli, ["update", "--project-dir", str(mock_project_dir)])

        assert_cli_success(result, context="Basic update command")
        mock_project_wrapper.assert_called_once()
        mock_project.update_pipelines.assert_called_once()

    @pytest.mark.integration
    def test_install_command_basic(self, mocker, runner, mock_project_dir):
        """Test basic install command functionality."""
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.install_pipelines.return_value = None
        mock_find_project.return_value = mock_project_dir

        result = runner.invoke(marimba_cli, ["install", "--project-dir", str(mock_project_dir)])

        assert_cli_success(result, context="Basic install command")
        mock_project_wrapper.assert_called_once()
        mock_project.install_pipelines.assert_called_once()

    @pytest.mark.integration
    def test_install_command_with_pipeline(self, mocker, runner, mock_project_dir):
        """Test install command with specific pipeline."""
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.install_pipelines.return_value = None
        mock_find_project.return_value = mock_project_dir

        result = runner.invoke(marimba_cli, ["install", "--project-dir", str(mock_project_dir)])

        assert_cli_success(result, context="Install command with pipeline")
        mock_project.install_pipelines.assert_called_once()

    @pytest.mark.integration
    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(marimba_cli, ["--help"])
        assert_cli_success(result, expected_message="Marimba", context="CLI help")
        assert "FAIR scientific" in result.stdout
        assert "image datasets" in result.stdout

    @pytest.mark.integration
    def test_cli_no_args(self, runner):
        """Test CLI with no arguments shows help."""
        result = runner.invoke(marimba_cli, [])
        # CLI exits with code 2 when no command is provided (missing required arguments)
        # This is expected behavior for Typer CLIs
        assert result.exit_code == 2
        # Should show usage information
        assert "Usage:" in result.stdout

    @pytest.mark.integration
    def test_import_command_help(self, runner):
        """Test import command help."""
        result = runner.invoke(marimba_cli, ["import", "--help"])
        assert_cli_success(result, context="Import command help")
        assert "source-path" in result.stdout or "SOURCE_PATH" in result.stdout

    @pytest.mark.integration
    def test_process_command_help(self, runner):
        """Test process command help."""
        result = runner.invoke(marimba_cli, ["process", "--help"])
        assert_cli_success(result, context="Process command help")
        assert "collection-name" in result.stdout or "COLLECTION_NAME" in result.stdout

    @pytest.mark.integration
    def test_package_command_help(self, runner):
        """Test package command help."""
        result = runner.invoke(marimba_cli, ["package", "--help"])
        assert_cli_success(result, context="Package command help")
        assert "collection-name" in result.stdout or "COLLECTION_NAME" in result.stdout

    @pytest.mark.integration
    def test_distribute_command_help(self, runner):
        """Test distribute command help."""
        result = runner.invoke(marimba_cli, ["distribute", "--help"])
        assert_cli_success(result, context="Distribute command help")
        assert "dataset-name" in result.stdout or "DATASET_NAME" in result.stdout

    @pytest.mark.integration
    def test_update_command_help(self, runner):
        """Test update command help."""
        result = runner.invoke(marimba_cli, ["update", "--help"])
        assert_cli_success(result, context="Update command help")

    @pytest.mark.integration
    def test_install_command_help(self, runner):
        """Test install command help."""
        result = runner.invoke(marimba_cli, ["install", "--help"])
        assert_cli_success(result, context="Install command help")


class TestCommandErrorHandling:
    """Test CLI command error handling."""

    @pytest.fixture
    def mock_project_dir(self, tmp_path):
        """Create a mock project directory."""
        return tmp_path / "test_project"

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.mark.integration
    def test_import_command_project_error(self, mocker, runner, mock_project_dir):
        """Test import command when project wrapper raises error."""
        mocker.patch("marimba.main.validate_dependencies")
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project_wrapper.side_effect = Exception("Project error")
        mock_find_project.return_value = mock_project_dir

        source_path = mock_project_dir / "source"
        source_path.mkdir(parents=True)

        result = runner.invoke(
            marimba_cli,
            ["import", "test_collection", str(source_path), "--project-dir", str(mock_project_dir)],
        )

        assert_cli_failure(result, expected_exit_code=1, context="Import command with project error")

    @pytest.mark.integration
    def test_process_command_processing_error(self, mocker, runner, mock_project_dir):
        """Test process command when processing raises error."""
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.run_process.side_effect = Exception("Processing error")
        mock_project.collection_wrappers = {"test_collection": mocker.Mock()}
        mock_project.pipeline_wrappers = {"test_pipeline": mocker.Mock()}
        mock_find_project.return_value = mock_project_dir

        result = runner.invoke(
            marimba_cli,
            [
                "process",
                "--collection-name",
                "test_collection",
                "--pipeline-name",
                "test_pipeline",
                "--project-dir",
                str(mock_project_dir),
            ],
        )

        # Command should show error message when processing fails
        assert "Error during processing: Processing error" in result.stdout

    @pytest.mark.integration
    def test_package_command_packaging_error(self, mocker, runner, mock_project_dir):
        """Test package command when packaging raises error."""
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.compose.side_effect = Exception("Packaging error")
        mock_project.collection_wrappers = {"test_collection": mocker.Mock()}
        mock_project.pipeline_wrappers = {"pipeline1": mocker.Mock()}
        mock_find_project.return_value = mock_project_dir

        result = runner.invoke(
            marimba_cli,
            ["package", "test_dataset", "--collection-name", "test_collection", "--project-dir", str(mock_project_dir)],
        )

        assert_cli_failure(result, expected_exit_code=1, context="Package command with packaging error")

    @pytest.mark.integration
    def test_distribute_command_distribution_error(self, mocker, runner, mock_project_dir):
        """Test distribute command when distribution raises error."""
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.distribute.side_effect = Exception("Distribution error")
        mock_find_project.return_value = mock_project_dir

        result = runner.invoke(
            marimba_cli,
            ["distribute", "test_dataset", "test_target", "--project-dir", str(mock_project_dir)],
        )

        assert_cli_failure(result, expected_exit_code=1, context="Distribute command with distribution error")


class TestCLIIntegration:
    """Test CLI integration scenarios."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.mark.integration
    def test_version_flag_in_global_options(self, mocker, runner):
        """Test --version flag in global options."""
        mocker.patch("importlib.metadata.version", return_value="1.0.0")
        result = runner.invoke(marimba_cli, ["--version"])
        assert_cli_success(result, expected_message="Marimba v1.0.0", context="Version flag in global options")

    @pytest.mark.integration
    def test_debug_flag(self, mocker, runner):
        """Test --debug flag."""
        mock_logger = mocker.patch("marimba.main.get_logger")
        mock_logger.return_value = mocker.Mock()
        _ = runner.invoke(marimba_cli, ["--debug", "version"])
        # Should not error with debug flag

    @pytest.mark.integration
    def test_quiet_flag(self, mocker, runner):
        """Test --quiet flag."""
        mock_logger = mocker.patch("marimba.main.get_logger")
        mock_logger.return_value = mocker.Mock()
        _ = runner.invoke(marimba_cli, ["--quiet", "version"])
        # Should not error with quiet flag

    @pytest.mark.integration
    def test_end_to_end_workflow_simulation(self, mocker, runner, tmp_path):
        """Test simulated end-to-end workflow commands."""
        # Set up mocks
        mocker.patch("marimba.main.validate_dependencies")
        mock_find_project = mocker.patch("marimba.main.find_project_dir_or_exit")
        mock_project_wrapper = mocker.patch("marimba.main.ProjectWrapper")
        mock_project = mocker.Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.run_import.return_value = None
        mock_project.run_process.return_value = None
        mock_project.compose.return_value = {"collection1": mocker.Mock()}
        mock_project.create_dataset.return_value = mocker.Mock()
        mock_project.collection_wrappers = {}  # Empty initially, will be populated after import
        mock_project.pipeline_wrappers = {"test_pipeline": mocker.Mock()}
        mock_project.prompt_collection_config.return_value = {"test": "config"}
        mock_project.create_collection.return_value = None
        mock_project.get_pipeline_post_processors.return_value = []

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        source_path = tmp_path / "source"
        source_path.mkdir()

        mock_find_project.return_value = project_dir

        # Simulate workflow: import -> process -> package
        # Import
        result1 = runner.invoke(
            marimba_cli,
            ["import", "test_collection", str(source_path), "--project-dir", str(project_dir)],
        )
        assert_cli_success(result1, context="End-to-end workflow import")

        # After import, collection should exist for process/package commands
        mock_project.collection_wrappers = {"test_collection": mocker.Mock()}

        # Process
        result2 = runner.invoke(
            marimba_cli,
            [
                "process",
                "--collection-name",
                "test_collection",
                "--pipeline-name",
                "test_pipeline",
                "--project-dir",
                str(project_dir),
            ],
        )
        assert_cli_success(result2, context="End-to-end workflow process")

        # Package
        result3 = runner.invoke(
            marimba_cli,
            ["package", "test_dataset", "--collection-name", "test_collection", "--project-dir", str(project_dir)],
        )
        assert_cli_success(result3, context="End-to-end workflow package")

        # Verify all operations were called
        mock_project.run_import.assert_called()
        mock_project.run_process.assert_called()
        mock_project.compose.assert_called()
        mock_project.create_dataset.assert_called()

    @pytest.mark.unit
    def test_subcommand_structure(self):
        """Test that subcommands are properly registered."""
        # Check that the CLI has the expected structure
        assert hasattr(marimba_cli, "registered_commands")

        # The CLI should have commands registered via decorators
        expected_commands = {"import", "package", "process", "distribute", "update", "install", "version"}
        actual_commands = {cmd.name for cmd in marimba_cli.registered_commands}

        assert expected_commands.issubset(actual_commands)
        assert len(marimba_cli.registered_commands) >= len(expected_commands)
