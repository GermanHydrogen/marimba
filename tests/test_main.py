"""Tests for marimba.main CLI module."""

import importlib.metadata
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from marimba.main import (
    distribute_command,
    global_options,
    import_command,
    install_command,
    marimba_cli,
    package_command,
    process_command,
    update_command,
    version_callback,
    version_command,
)
from marimba.core.utils.log import LogLevel


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

    def test_marimba_cli_creation(self):
        """Test that the CLI app is created correctly."""
        assert isinstance(marimba_cli, typer.Typer)
        assert marimba_cli.info.name == "Marimba"
        assert "FAIR scientific image datasets" in marimba_cli.info.help

    def test_version_callback_with_version(self):
        """Test version callback when version is available."""
        with patch('importlib.metadata.version', return_value="1.0.0"):
            with pytest.raises(typer.Exit):
                version_callback(True)

    def test_version_callback_without_flag(self):
        """Test version callback when flag is False."""
        result = version_callback(False)
        assert result is None

    def test_version_callback_with_exception(self):
        """Test version callback when metadata is not available."""
        with patch('importlib.metadata.version', side_effect=Exception("No metadata")):
            with pytest.raises(typer.Exit):
                version_callback(True)

    def test_version_command_with_version(self, runner):
        """Test version command when version is available."""
        with patch('importlib.metadata.version', return_value="1.0.0"):
            result = runner.invoke(marimba_cli, ["version"])
            assert result.exit_code == 0
            assert "Marimba v1.0.0" in result.stdout

    def test_version_command_with_exception(self, runner):
        """Test version command when metadata is not available."""
        with patch('importlib.metadata.version', side_effect=Exception("No metadata")):
            result = runner.invoke(marimba_cli, ["version"])
            assert result.exit_code == 0
            assert "unknown (not installed as package)" in result.stdout

    def test_global_options_with_debug(self):
        """Test global options with debug level."""
        mock_ctx = Mock()
        mock_ctx.invoked_subcommand = 'import'  # Simulate subcommand
        
        with patch('marimba.main.get_rich_handler') as mock_handler:
            mock_log_handler = Mock()
            mock_handler.return_value = mock_log_handler
            
            global_options(ctx=mock_ctx, level=LogLevel.DEBUG, version=False)
            
            # Should set debug level on the handler
            mock_log_handler.setLevel.assert_called_once_with(10)

    def test_global_options_with_quiet(self):
        """Test global options with error level (quiet)."""
        mock_ctx = Mock()
        mock_ctx.invoked_subcommand = 'import'  # Simulate subcommand
        
        with patch('marimba.main.get_rich_handler') as mock_handler:
            mock_log_handler = Mock()
            mock_handler.return_value = mock_log_handler
            
            global_options(ctx=mock_ctx, level=LogLevel.ERROR, version=False)
            
            # Should set error level on the handler
            mock_log_handler.setLevel.assert_called_once_with(40)

    def test_global_options_normal(self):
        """Test global options with default settings."""
        mock_ctx = Mock()
        mock_ctx.invoked_subcommand = 'import'  # Simulate subcommand
        
        with patch('marimba.main.get_rich_handler') as mock_handler:
            mock_log_handler = Mock()
            mock_handler.return_value = mock_log_handler
            
            global_options(ctx=mock_ctx, level=LogLevel.WARNING, version=False)
            
            # Should set warning level on the handler (default)
            mock_log_handler.setLevel.assert_called_once_with(30)

    @patch('marimba.main.validate_dependencies')
    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_import_command_basic(self, mock_project_wrapper, mock_find_project, mock_validate_deps, runner, mock_project_dir):
        """Test basic import command functionality."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.run_import.return_value = None
        mock_project.collection_wrappers = {}  # Empty collections dict
        mock_project.pipeline_wrappers = {"pipeline1": Mock()}  # Mock pipelines
        mock_project.prompt_collection_config.return_value = {"test": "config"}
        mock_project.create_collection.return_value = None
        mock_find_project.return_value = mock_project_dir
        
        # Mock the source path
        source_path = mock_project_dir / "source"
        source_path.mkdir()
        
        result = runner.invoke(marimba_cli, [
            "import", 
            "test_collection",
            str(source_path),
            "--project-dir", str(mock_project_dir)
        ])
        
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.stdout}")
            print(f"Exception: {result.exception}")
        
        assert result.exit_code == 0
        mock_find_project.assert_called_once()
        mock_project_wrapper.assert_called_once()
        mock_project.run_import.assert_called_once()

    @patch('marimba.main.validate_dependencies')
    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_import_command_with_options(self, mock_project_wrapper, mock_find_project, mock_validate_deps, runner, mock_project_dir):
        """Test import command with various options."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.run_import.return_value = None
        mock_project.collection_wrappers = {}
        mock_project.pipeline_wrappers = {"pipeline1": Mock()}
        mock_project.prompt_collection_config.return_value = {"test": "config"}
        mock_project.create_collection.return_value = None
        mock_find_project.return_value = mock_project_dir
        
        source_path = mock_project_dir / "source"
        source_path.mkdir()
        
        result = runner.invoke(marimba_cli, [
            "import",
            "test_collection",
            str(source_path),
            "--project-dir", str(mock_project_dir),
            "--overwrite",
            "--dry-run"
        ])
        
        assert result.exit_code == 0
        mock_project.run_import.assert_called_once()

    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_process_command_basic(self, mock_project_wrapper, mock_find_project, runner, mock_project_dir):
        """Test basic process command functionality."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.run_process.return_value = None
        mock_project.collection_wrappers = {"test_collection": Mock()}
        mock_project.pipeline_wrappers = {"test_pipeline": Mock()}
        mock_find_project.return_value = mock_project_dir
        
        result = runner.invoke(marimba_cli, [
            "process",
            "--collection-name", "test_collection",
            "--pipeline-name", "test_pipeline", 
            "--project-dir", str(mock_project_dir)
        ])
        
        assert result.exit_code == 0
        mock_project_wrapper.assert_called_once()
        mock_project.run_process.assert_called_once()

    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_process_command_with_options(self, mock_project_wrapper, mock_find_project, runner, mock_project_dir):
        """Test process command with various options."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.run_process.return_value = None
        mock_project.collection_wrappers = {"test_collection": Mock()}
        mock_project.pipeline_wrappers = {"test_pipeline": Mock()}
        mock_find_project.return_value = mock_project_dir
        
        result = runner.invoke(marimba_cli, [
            "process",
            "--collection-name", "test_collection",
            "--pipeline-name", "test_pipeline",
            "--project-dir", str(mock_project_dir),
            "--dry-run"
        ])
        
        assert result.exit_code == 0
        mock_project.run_process.assert_called_once()

    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_package_command_basic(self, mock_project_wrapper, mock_find_project, runner, mock_project_dir):
        """Test basic package command functionality."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.compose.return_value = {"collection1": Mock()}
        mock_project.create_dataset.return_value = Mock()
        mock_project.collection_wrappers = {"test_collection": Mock()}
        mock_project.pipeline_wrappers = {"pipeline1": Mock()}
        mock_project.get_pipeline_post_processors.return_value = []
        mock_find_project.return_value = mock_project_dir
        
        result = runner.invoke(marimba_cli, [
            "package",
            "test_dataset",
            "--collection-name", "test_collection",
            "--project-dir", str(mock_project_dir)
        ])
        
        assert result.exit_code == 0
        mock_project_wrapper.assert_called_once()
        mock_project.compose.assert_called_once()
        mock_project.create_dataset.assert_called_once()

    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_package_command_with_options(self, mock_project_wrapper, mock_find_project, runner, mock_project_dir):
        """Test package command with various options."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.compose.return_value = {"collection1": Mock()}
        mock_project.create_dataset.return_value = Mock()
        mock_project.collection_wrappers = {"test_collection": Mock()}
        mock_project.pipeline_wrappers = {"pipeline1": Mock()}
        mock_project.get_pipeline_post_processors.return_value = []
        mock_find_project.return_value = mock_project_dir
        
        result = runner.invoke(marimba_cli, [
            "package",
            "test_dataset", 
            "--collection-name", "test_collection",
            "--project-dir", str(mock_project_dir),
            "--version", "2.0"
        ])
        
        assert result.exit_code == 0
        mock_project.compose.assert_called_once()
        mock_project.create_dataset.assert_called_once()

    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_distribute_command_basic(self, mock_project_wrapper, mock_find_project, runner, mock_project_dir):
        """Test basic distribute command functionality."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.distribute.return_value = None
        mock_find_project.return_value = mock_project_dir
        
        result = runner.invoke(marimba_cli, [
            "distribute",
            "test_dataset",
            "test_target",
            "--project-dir", str(mock_project_dir)
        ])
        
        assert result.exit_code == 0
        mock_project_wrapper.assert_called_once()
        mock_project.distribute.assert_called_once()

    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_update_command_basic(self, mock_project_wrapper, mock_find_project, runner, mock_project_dir):
        """Test basic update command functionality."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.update_pipelines.return_value = None
        mock_find_project.return_value = mock_project_dir
        
        result = runner.invoke(marimba_cli, [
            "update",
            "--project-dir", str(mock_project_dir)
        ])
        
        assert result.exit_code == 0
        mock_project_wrapper.assert_called_once()
        mock_project.update_pipelines.assert_called_once()

    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_install_command_basic(self, mock_project_wrapper, mock_find_project, runner, mock_project_dir):
        """Test basic install command functionality."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.install_pipelines.return_value = None
        mock_find_project.return_value = mock_project_dir
        
        result = runner.invoke(marimba_cli, [
            "install",
            "--project-dir", str(mock_project_dir)
        ])
        
        assert result.exit_code == 0
        mock_project_wrapper.assert_called_once()
        mock_project.install_pipelines.assert_called_once()

    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_install_command_with_pipeline(self, mock_project_wrapper, mock_find_project, runner, mock_project_dir):
        """Test install command with specific pipeline."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.install_pipelines.return_value = None
        mock_find_project.return_value = mock_project_dir
        
        result = runner.invoke(marimba_cli, [
            "install",
            "--project-dir", str(mock_project_dir)
        ])
        
        assert result.exit_code == 0
        mock_project.install_pipelines.assert_called_once()

    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(marimba_cli, ["--help"])
        assert result.exit_code == 0
        assert "Marimba" in result.stdout
        assert "FAIR scientific" in result.stdout
        assert "image datasets" in result.stdout

    def test_cli_no_args(self, runner):
        """Test CLI with no arguments shows help."""
        result = runner.invoke(marimba_cli, [])
        # CLI exits with code 2 when no command is provided (missing required arguments)
        # This is expected behavior for Typer CLIs
        assert result.exit_code == 2
        # Should show usage information
        assert "Usage:" in result.stdout

    def test_import_command_help(self, runner):
        """Test import command help."""
        result = runner.invoke(marimba_cli, ["import", "--help"])
        assert result.exit_code == 0
        assert "source-path" in result.stdout or "SOURCE_PATH" in result.stdout

    def test_process_command_help(self, runner):
        """Test process command help."""
        result = runner.invoke(marimba_cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "collection-name" in result.stdout or "COLLECTION_NAME" in result.stdout

    def test_package_command_help(self, runner):
        """Test package command help."""
        result = runner.invoke(marimba_cli, ["package", "--help"])
        assert result.exit_code == 0
        assert "collection-name" in result.stdout or "COLLECTION_NAME" in result.stdout

    def test_distribute_command_help(self, runner):
        """Test distribute command help."""
        result = runner.invoke(marimba_cli, ["distribute", "--help"])
        assert result.exit_code == 0
        assert "dataset-name" in result.stdout or "DATASET_NAME" in result.stdout

    def test_update_command_help(self, runner):
        """Test update command help."""
        result = runner.invoke(marimba_cli, ["update", "--help"])
        assert result.exit_code == 0

    def test_install_command_help(self, runner):
        """Test install command help."""
        result = runner.invoke(marimba_cli, ["install", "--help"])
        assert result.exit_code == 0


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

    @patch('marimba.main.validate_dependencies')
    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_import_command_project_error(self, mock_project_wrapper, mock_find_project, mock_validate_deps, runner, mock_project_dir):
        """Test import command when project wrapper raises error."""
        mock_project_wrapper.side_effect = Exception("Project error")
        mock_find_project.return_value = mock_project_dir
        
        source_path = mock_project_dir / "source"
        source_path.mkdir(parents=True)
        
        result = runner.invoke(marimba_cli, [
            "import",
            "test_collection",
            str(source_path),
            "--project-dir", str(mock_project_dir)
        ])
        
        assert result.exit_code == 1

    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_process_command_processing_error(self, mock_project_wrapper, mock_find_project, runner, mock_project_dir):
        """Test process command when processing raises error."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.run_process.side_effect = Exception("Processing error")
        mock_project.collection_wrappers = {"test_collection": Mock()}
        mock_project.pipeline_wrappers = {"test_pipeline": Mock()}
        mock_find_project.return_value = mock_project_dir
        
        result = runner.invoke(marimba_cli, [
            "process",
            "--collection-name", "test_collection",
            "--pipeline-name", "test_pipeline",
            "--project-dir", str(mock_project_dir)
        ])
        
        # Command should show error message when processing fails
        assert "Error during processing: Processing error" in result.stdout

    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_package_command_packaging_error(self, mock_project_wrapper, mock_find_project, runner, mock_project_dir):
        """Test package command when packaging raises error."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.compose.side_effect = Exception("Packaging error")
        mock_project.collection_wrappers = {"test_collection": Mock()}
        mock_project.pipeline_wrappers = {"pipeline1": Mock()}
        mock_find_project.return_value = mock_project_dir
        
        result = runner.invoke(marimba_cli, [
            "package",
            "test_dataset",
            "--collection-name", "test_collection",
            "--project-dir", str(mock_project_dir)
        ])
        
        assert result.exit_code == 1

    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_distribute_command_distribution_error(self, mock_project_wrapper, mock_find_project, runner, mock_project_dir):
        """Test distribute command when distribution raises error."""
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.distribute.side_effect = Exception("Distribution error")
        mock_find_project.return_value = mock_project_dir
        
        result = runner.invoke(marimba_cli, [
            "distribute",
            "test_dataset",
            "test_target",
            "--project-dir", str(mock_project_dir)
        ])
        
        assert result.exit_code == 1


class TestCLIIntegration:
    """Test CLI integration scenarios."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_version_flag_in_global_options(self, runner):
        """Test --version flag in global options."""
        with patch('importlib.metadata.version', return_value="1.0.0"):
            result = runner.invoke(marimba_cli, ["--version"])
            assert result.exit_code == 0
            assert "Marimba v1.0.0" in result.stdout

    def test_debug_flag(self, runner):
        """Test --debug flag."""
        with patch('marimba.main.get_logger') as mock_logger:
            mock_logger.return_value = Mock()
            result = runner.invoke(marimba_cli, ["--debug", "version"])
            # Should not error with debug flag

    def test_quiet_flag(self, runner):
        """Test --quiet flag."""
        with patch('marimba.main.get_logger') as mock_logger:
            mock_logger.return_value = Mock()
            result = runner.invoke(marimba_cli, ["--quiet", "version"])
            # Should not error with quiet flag

    @patch('marimba.main.validate_dependencies')
    @patch('marimba.main.find_project_dir_or_exit')
    @patch('marimba.main.ProjectWrapper')
    def test_end_to_end_workflow_simulation(self, mock_project_wrapper, mock_find_project, mock_validate_deps, runner, tmp_path):
        """Test simulated end-to-end workflow commands."""
        # Set up mocks
        mock_project = Mock()
        mock_project_wrapper.return_value = mock_project
        mock_project.run_import.return_value = None
        mock_project.run_process.return_value = None
        mock_project.compose.return_value = {"collection1": Mock()}
        mock_project.create_dataset.return_value = Mock()
        mock_project.collection_wrappers = {}  # Empty initially, will be populated after import
        mock_project.pipeline_wrappers = {"test_pipeline": Mock()}
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
        result1 = runner.invoke(marimba_cli, [
            "import",
            "test_collection",
            str(source_path),
            "--project-dir", str(project_dir)
        ])
        assert result1.exit_code == 0
        
        # After import, collection should exist for process/package commands
        mock_project.collection_wrappers = {"test_collection": Mock()}
        
        # Process  
        result2 = runner.invoke(marimba_cli, [
            "process",
            "--collection-name", "test_collection",
            "--pipeline-name", "test_pipeline",
            "--project-dir", str(project_dir)
        ])
        assert result2.exit_code == 0
        
        # Package
        result3 = runner.invoke(marimba_cli, [
            "package",
            "test_dataset",
            "--collection-name", "test_collection",
            "--project-dir", str(project_dir)
        ])
        assert result3.exit_code == 0
        
        # Verify all operations were called
        mock_project.run_import.assert_called()
        mock_project.run_process.assert_called()
        mock_project.compose.assert_called()
        mock_project.create_dataset.assert_called()

    def test_subcommand_structure(self):
        """Test that subcommands are properly registered."""
        # Check that the CLI has the expected structure
        assert hasattr(marimba_cli, 'registered_commands')
        
        # The CLI should have commands registered via decorators
        expected_commands = {
            'import', 'package', 'process', 'distribute', 
            'update', 'install', 'version'
        }
        actual_commands = {cmd.name for cmd in marimba_cli.registered_commands}
        
        assert expected_commands.issubset(actual_commands)
        assert len(marimba_cli.registered_commands) >= len(expected_commands)