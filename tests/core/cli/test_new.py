import os
from pathlib import Path

import pytest
from pytest_mock import MockerFixture
from typer.testing import CliRunner

from marimba.core.wrappers.project import ProjectWrapper
from marimba.main import marimba_cli
from tests.conftest import (
    assert_project_structure_complete,
    run_cli_command,
)

runner = CliRunner()

# ---------------------------------------------------------------------------------------------------------------------#
# Testing project()
# ---------------------------------------------------------------------------------------------------------------------#


class TestProjectCommand:
    """Test class for the project CLI command."""

    @pytest.mark.integration
    def test_project_creates_new_project_successfully(self, tmp_path: Path) -> None:
        """
        Test project command creates a new Marimba project successfully.

        This integration test verifies that the CLI command properly orchestrates
        project creation by testing the real interaction between CLI argument parsing,
        ProjectWrapper.create(), and filesystem operations.
        """
        # Arrange
        project_dir = tmp_path / "new_project"

        # Act - Test real project creation without excessive mocking
        run_cli_command(
            runner,
            ["new", "project", str(project_dir)],
            expected_success=True,
            expected_message="Created new Marimba project at",
            context="Project creation",
        )

        # Assert - Verify real project structure was created
        assert_project_structure_complete(project_dir, "New project creation")

    @pytest.mark.unit
    def test_project_exits_if_project_exists(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test project command exits with error when project directory already exists.

        This unit test verifies that when ProjectWrapper.create() raises FileExistsError,
        the CLI command exits with code 1 and displays the error message
        "A Marimba project already exists at:" followed by the project directory path.
        """
        # Arrange - Set up existing project directory and mock ProjectWrapper.create() to fail
        existing_project_dir = tmp_path / "existing_project"
        existing_project_dir.mkdir()

        mock_project_create = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create",
            side_effect=FileExistsError("Project exists"),
        )

        # Act & Assert - Using shared CLI failure helper for consistent error checking
        run_cli_command(
            runner,
            ["new", "project", str(existing_project_dir)],
            expected_success=False,
            expected_message="A Marimba project already exists at:",
            context="Project already exists",
        )

        # Assert - Verify the correct method was called with the expected directory
        mock_project_create.assert_called_once_with(existing_project_dir)

    @pytest.mark.integration
    def test_project_exit_code_on_success(self, tmp_path: Path) -> None:
        """
        Test project command exits with code 0 on successful project creation.

        This integration test verifies that the CLI command returns the correct
        exit code (0) when project creation succeeds, testing the real interaction
        between CLI command execution and success handling.
        """
        # Arrange
        project_dir = tmp_path / "success_test_project"

        # Act
        result = runner.invoke(marimba_cli, ["new", "project", str(project_dir)])

        # Assert
        assert result.exit_code == 0, f"Command should exit with code 0 on success, got {result.exit_code}"
        assert "Created new Marimba project at" in result.output, "Success message should be displayed"
        # Rich formatting may wrap long paths, so check for key components of the project name
        assert "success_te" in result.output, "First part of project name should be displayed in success message"
        assert "st_project" in result.output, "Second part of project name should be displayed in success message"

        # Verify project was actually created with proper structure
        assert_project_structure_complete(project_dir, "Project creation with success message")

    @pytest.mark.integration
    def test_project_handles_existing_directory_path(self, tmp_path: Path) -> None:
        """
        Test project command handles existing directory path with appropriate error message.

        This integration test verifies that the CLI command properly handles cases where the
        specified project directory already exists, ensuring the user receives
        a clear error message about the existing project and the command exits with code 1.
        """
        # Arrange - Create directory that already exists
        existing_project_dir = tmp_path / "existing_project"
        existing_project_dir.mkdir()

        # Act
        result = runner.invoke(marimba_cli, ["new", "project", str(existing_project_dir)])

        # Assert
        assert result.exit_code == 1, f"Command should fail with exit code 1, got {result.exit_code}"
        assert (
            "A Marimba project already exists at:" in result.output
        ), f"Should show project exists error, got: {result.output}"
        # Rich formatting may wrap long paths, so check for key components of the project name
        assert "existing_p" in result.output, "First part of project name should be in error message"
        assert "roject" in result.output, "Second part of project name should be in error message"

    @pytest.mark.unit
    def test_project_handles_creation_failure_with_proper_exit_code(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """
        Test project command exits with code 1 when ProjectWrapper.create() fails.

        This unit test verifies that when ProjectWrapper.create() raises FileExistsError,
        the CLI command exits with the correct error code and displays the appropriate
        error message. Uses the established run_cli_command helper for consistency.
        """
        # Arrange
        project_dir = tmp_path / "creation_failure_project"

        mock_create = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create",
            side_effect=FileExistsError("Project exists"),
        )

        # Act & Assert - Using shared CLI failure helper for consistent error checking
        run_cli_command(
            runner,
            ["new", "project", str(project_dir)],
            expected_success=False,
            expected_message="A Marimba project already exists at:",
            context="ProjectWrapper.create() failure handling",
        )

        # Verify the correct method was called with expected arguments
        mock_create.assert_called_once_with(project_dir)


# ---------------------------------------------------------------------------------------------------------------------#
# Testing pipeline()
# ---------------------------------------------------------------------------------------------------------------------#


class TestPipelineCommand:
    """Test class for the pipeline CLI command."""

    @pytest.mark.integration
    def test_pipeline_creates_new_pipeline_successfully(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test pipeline command creates a new pipeline successfully.

        This integration test verifies that the CLI command properly orchestrates
        pipeline creation by testing the interaction between CLI argument parsing,
        project directory resolution, and pipeline wrapper creation with minimal mocking
        of external dependencies only.
        """
        # Arrange
        project_dir = tmp_path / "project"
        pipeline_name = "test_pipeline"
        url = "https://example.com/repo.git"

        # Create a real Marimba project structure to test against
        ProjectWrapper.create(project_dir)

        mock_pipeline_wrapper = mocker.MagicMock()
        mock_pipeline_wrapper.root_dir = project_dir / "pipelines" / pipeline_name

        # Mock only external dependencies - the pipeline creation which involves Git operations
        mock_create_pipeline = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_pipeline",
            return_value=mock_pipeline_wrapper,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "pipeline", pipeline_name, url, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 0, f"Command should succeed, got output: {result.output}"
        assert "Created new Marimba pipeline" in result.output, "Success message should be displayed"
        assert f'"{pipeline_name}"' in result.output, "Pipeline name should be displayed in success message"
        mock_create_pipeline.assert_called_once_with(pipeline_name, url, {})

    @pytest.mark.unit
    def test_pipeline_invalid_name_error(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test pipeline command exits with error for invalid pipeline name.

        This unit test verifies proper error handling when an invalid pipeline
        name causes ProjectWrapper.create_pipeline() to raise InvalidNameError.
        """
        # Arrange
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        pipeline_name = "invalid/name"
        url = "https://example.com/repo.git"
        expected_error_message = "Invalid name"

        mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_pipeline",
            side_effect=ProjectWrapper.InvalidNameError(expected_error_message),
        )
        mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "pipeline", pipeline_name, url, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 1, f"Command should exit with error code 1, got {result.exit_code}"
        assert "Invalid pipeline name:" in result.output, f"Should show invalid name error, got: {result.output}"
        assert (
            expected_error_message in result.output
        ), f"Should include specific error details '{expected_error_message}', got: {result.output}"

    @pytest.mark.unit
    def test_pipeline_creation_failure(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test pipeline command exits with error when pipeline creation fails.

        This unit test verifies proper error handling for general exceptions
        during pipeline creation, ensuring the CLI displays appropriate error messages.
        """
        # Arrange
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        pipeline_name = "test_pipeline"
        url = "https://example.com/repo.git"
        expected_error_message = "Creation failed"

        mock_create_pipeline = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_pipeline",
            side_effect=Exception(expected_error_message),
        )
        mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "pipeline", pipeline_name, url, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 1, f"Command should exit with error code 1, got {result.exit_code}"
        assert (
            "Could not create pipeline:" in result.output
        ), f"Should show creation error message, got: {result.output}"
        assert (
            expected_error_message in result.output
        ), f"Should include specific error details '{expected_error_message}', got: {result.output}"

        # Verify the correct method was called with expected arguments
        mock_create_pipeline.assert_called_once_with(pipeline_name, url, {})

    @pytest.mark.unit
    def test_pipeline_logs_command_execution(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test pipeline command logs command execution properly.

        This unit test verifies that the CLI command properly logs execution
        information when creating a new pipeline, focusing on testing the logging
        functionality in isolation with mocked dependencies.
        """
        # Arrange
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        pipeline_name = "test_pipeline"
        url = "https://example.com/repo.git"

        # Mock external dependencies and project directory detection
        mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)

        mock_pipeline_wrapper = mocker.MagicMock()
        mock_pipeline_wrapper.root_dir = project_dir / "pipelines" / pipeline_name

        # Mock pipeline creation to focus on logging behavior
        mock_create_pipeline = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_pipeline",
            return_value=mock_pipeline_wrapper,
        )

        # Mock logger to capture logging calls
        mock_logger = mocker.patch("marimba.core.cli.new.logger")

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "pipeline", pipeline_name, url, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 0, f"Command should succeed, got output: {result.output}"
        assert "Created new Marimba pipeline" in result.output, "Should show success message"
        assert f'"{pipeline_name}"' in result.output, "Should include pipeline name in output"

        # Verify logging behavior - should log command execution with specific message
        mock_logger.info.assert_called_once()
        log_call_args = mock_logger.info.call_args[0][0]
        assert "Executing the" in log_call_args, "Should log command execution start"
        assert "new pipeline" in log_call_args, "Should include specific command name in log message"

        # Verify the create_pipeline method was called with expected arguments
        mock_create_pipeline.assert_called_once_with(pipeline_name, url, {})


# ---------------------------------------------------------------------------------------------------------------------#
# Testing collection()
# ---------------------------------------------------------------------------------------------------------------------#


class TestCollectionCommand:
    """Test class for the collection CLI command."""

    @pytest.mark.integration
    def test_collection_creates_new_collection(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test collection command creates a new collection successfully.

        This integration test verifies that the CLI command properly orchestrates collection creation
        by testing the real interaction between CLI argument parsing, project wrapper initialization,
        and collection configuration with minimal mocking of external dependencies only.
        """
        # Arrange
        project_dir = tmp_path / "test_project"
        collection_name = "test_collection"
        parent_collection_name = "parent_collection"

        # Create a real Marimba project structure for integration testing
        ProjectWrapper.create(project_dir)

        # Mock only external dependencies that involve user interaction
        mock_collection_config = {"parent": parent_collection_name, "metadata_schema": "ifdo"}
        mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.prompt_collection_config",
            return_value=mock_collection_config,
        )

        # Mock the create_collection method to avoid actual filesystem operations but test real wrapper creation
        mock_collection_wrapper = mocker.MagicMock()
        mock_collection_wrapper.root_dir = project_dir / "collections" / collection_name
        mock_create = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_collection",
            return_value=mock_collection_wrapper,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            [
                "new",
                "collection",
                collection_name,
                parent_collection_name,
                "--project-dir",
                str(project_dir),
            ],
        )

        # Assert
        assert (
            result.exit_code == 0
        ), f"Collection creation should succeed with exit code 0, got {result.exit_code} with output: {result.output}"
        assert (
            "Created new Marimba collection" in result.output
        ), f"Success message should be displayed in output, got: {result.output}"
        assert (
            f'"{collection_name}"' in result.output
        ), f"Collection name '{collection_name}' should appear in success message, got: {result.output}"

        # Verify the create_collection was called with correct arguments from CLI parsing
        mock_create.assert_called_once_with(
            collection_name,
            mock_collection_config,
        ), "ProjectWrapper.create_collection should be called with parsed CLI arguments and prompted config"

    @pytest.mark.unit
    def test_collection_invalid_name_error(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test collection command exits with error for invalid collection name.

        This unit test verifies proper error handling when an invalid collection name
        causes ProjectWrapper.create_collection() to raise InvalidNameError, ensuring
        the CLI displays appropriate error messages and exits with code 1.
        """
        # Arrange
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        collection_name = "invalid/name"
        expected_error_message = "Invalid name"
        mock_collection_config = {"parent": None, "metadata_schema": "ifdo"}

        # Mock external dependencies - project location, configuration prompt, and collection creation
        mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
        mock_prompt_config = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.prompt_collection_config",
            return_value=mock_collection_config,
        )
        mock_create_collection = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_collection",
            side_effect=ProjectWrapper.InvalidNameError(expected_error_message),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "collection", collection_name, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 1, f"Command should exit with error code 1, got {result.exit_code}"
        assert (
            "Invalid collection name:" in result.output
        ), f"Should show invalid name error message, got: {result.output}"
        assert (
            expected_error_message in result.output
        ), f"Should include specific error details '{expected_error_message}', got: {result.output}"

        # Verify the correct methods were called with expected arguments
        mock_prompt_config.assert_called_once_with(parent_collection_name=None, config={})
        mock_create_collection.assert_called_once_with(collection_name, mock_collection_config)

    @pytest.mark.unit
    def test_collection_no_such_parent_collection_error(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """
        Test collection command exits with error when parent collection does not exist.

        This unit test verifies proper error handling when ProjectWrapper.create_collection()
        raises NoSuchCollectionError for a non-existent parent collection, ensuring the CLI
        displays appropriate error messages and exits with code 1. Uses mocking to isolate
        the error handling logic from external dependencies.
        """
        # Arrange
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        collection_name = "test_collection"
        parent_collection_name = "non_existent_parent"
        expected_error_message = "No such parent collection"
        mock_collection_config = {"parent": parent_collection_name, "metadata_schema": "ifdo"}

        # Mock external dependencies - project location, configuration prompt, and collection creation
        mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
        mock_prompt_config = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.prompt_collection_config",
            return_value=mock_collection_config,
        )
        mock_create_collection = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_collection",
            side_effect=ProjectWrapper.NoSuchCollectionError(expected_error_message),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            [
                "new",
                "collection",
                collection_name,
                parent_collection_name,
                "--project-dir",
                str(project_dir),
            ],
        )

        # Assert
        assert result.exit_code == 1, f"Command should exit with error code 1, got {result.exit_code}"
        assert (
            "No such parent collection:" in result.output
        ), f"Should display parent collection error message, got: {result.output}"
        assert (
            expected_error_message in result.output
        ), f"Should include specific error details '{expected_error_message}', got: {result.output}"

        # Verify the correct methods were called with expected arguments
        mock_prompt_config.assert_called_once_with(parent_collection_name=parent_collection_name, config={})
        mock_create_collection.assert_called_once_with(collection_name, mock_collection_config)

    @pytest.mark.unit
    def test_collection_creation_specific_failure(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test collection command exits with error when collection creation fails with CreateCollectionError.

        This unit test verifies proper error handling for CreateCollectionError exceptions,
        ensuring the CLI displays appropriate error messages and exits with code 1.
        Uses mocking to isolate the error handling logic from external dependencies.
        """
        # Arrange
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        collection_name = "test_collection"
        expected_error_message = "Creation failed"
        mock_collection_config = {"parent": None, "metadata_schema": "ifdo"}

        mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
        mock_prompt_config = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.prompt_collection_config",
            return_value=mock_collection_config,
        )
        mock_create_collection = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_collection",
            side_effect=ProjectWrapper.CreateCollectionError(expected_error_message),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "collection", collection_name, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 1, f"Command should exit with error code 1, got {result.exit_code}"
        assert (
            "Could not create collection:" in result.output
        ), f"Should show creation error message, got: {result.output}"
        assert (
            expected_error_message in result.output
        ), f"Should include specific error details '{expected_error_message}', got: {result.output}"

        # Verify the correct methods were called with expected arguments
        mock_prompt_config.assert_called_once_with(parent_collection_name=None, config={})
        mock_create_collection.assert_called_once_with(collection_name, mock_collection_config)

    @pytest.mark.unit
    def test_collection_creation_general_failure(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test collection command exits with error when collection creation fails with general exception.

        This unit test verifies proper error handling for unexpected exceptions during collection creation,
        ensuring the CLI displays appropriate error messages and exits with code 1. Uses mocking to
        isolate the error handling logic from external dependencies.
        """
        # Arrange
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        collection_name = "test_collection"
        expected_error_message = "Unexpected error occurred"
        mock_collection_config = {"parent": None, "metadata_schema": "ifdo"}

        mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
        mock_prompt_config = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.prompt_collection_config",
            return_value=mock_collection_config,
        )
        mock_create_collection = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_collection",
            side_effect=Exception(expected_error_message),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "collection", collection_name, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 1, f"Command should exit with error code 1, got {result.exit_code}"
        assert (
            "Could not create collection:" in result.output
        ), f"Should show general error message, got: {result.output}"
        assert (
            expected_error_message in result.output
        ), f"Should include specific error details '{expected_error_message}', got: {result.output}"

        # Verify the correct methods were called with expected arguments
        mock_prompt_config.assert_called_once_with(parent_collection_name=None, config={})
        mock_create_collection.assert_called_once_with(collection_name, mock_collection_config)

    @pytest.mark.integration
    def test_collection_creation_other_failure(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test collection command exits with error when collection creation fails with general exception.

        This integration test verifies proper error handling for unexpected exceptions during
        collection creation, testing real component interactions with minimal mocking of
        external dependencies only. Creates a real project structure and tests the integration
        between CLI parsing, project wrapper creation, and error handling.
        """
        # Arrange
        project_dir = tmp_path / "test_project"
        collection_name = "test_collection"
        expected_error_message = "Unexpected integration error"

        # Create a real Marimba project structure for integration testing
        ProjectWrapper.create(project_dir)

        # Mock only the external user interaction dependency - the configuration prompt
        mock_collection_config = {"parent": None, "metadata_schema": "ifdo"}
        mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.prompt_collection_config",
            return_value=mock_collection_config,
        )

        # Mock only the collection creation that would involve filesystem operations
        # to inject the failure, testing the integration between CLI and error handling
        mock_create_collection = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_collection",
            side_effect=Exception(expected_error_message),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "collection", collection_name, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 1, f"Command should exit with error code 1, got {result.exit_code}"
        assert (
            "Could not create collection:" in result.output
        ), f"Should show general error message, got: {result.output}"
        assert (
            expected_error_message in result.output
        ), f"Should include specific error details '{expected_error_message}', got: {result.output}"

        # Verify the correct methods were called with expected arguments
        # Tests real component interaction - CLI parsing arguments correctly
        mock_create_collection.assert_called_once_with(collection_name, mock_collection_config)


# ---------------------------------------------------------------------------------------------------------------------#
# Testing target()
# ---------------------------------------------------------------------------------------------------------------------#


class TestTargetCommand:
    """Test class for the target CLI command."""

    @pytest.mark.unit
    def test_target_creates_new_target_successfully(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test target command creates a new distribution target successfully.

        This unit test verifies that the CLI command properly orchestrates
        distribution target creation by calling ProjectWrapper.create_target()
        with the correct arguments and displays the expected success message
        including target name and configuration path.
        """
        # Arrange
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        target_name = "test_target"

        mock_target_wrapper = mocker.MagicMock()
        mock_target_wrapper.config_path = project_dir / "targets" / f"{target_name}.yml"

        mock_create_target = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_target",
            return_value=mock_target_wrapper,
        )
        mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
        mocker.patch(
            "marimba.core.wrappers.target.DistributionTargetWrapper.prompt_target",
            return_value=("target_type", "target_config"),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "target", target_name, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 0, f"Command should succeed, got output: {result.output}"
        mock_create_target.assert_called_once_with(target_name, "target_type", "target_config")
        assert "Created new Marimba target" in result.output, "Success message should be displayed"
        assert f'"{target_name}"' in result.output, "Target name should be displayed in success message"
        assert f"{target_name}.yml" in result.output, "Config path should be displayed in success message"

    @pytest.mark.unit
    def test_target_invalid_name_error(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test target command exits with error for invalid target name.

        This unit test verifies proper error handling when an invalid target
        name causes ProjectWrapper.create_target() to raise InvalidNameError.
        """
        # Arrange
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        target_name = "invalid/name"
        expected_error_message = "Invalid name"

        mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_target",
            side_effect=ProjectWrapper.InvalidNameError(expected_error_message),
        )
        mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
        mocker.patch(
            "marimba.core.wrappers.target.DistributionTargetWrapper.prompt_target",
            return_value=("target_type", "target_config"),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "target", target_name, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 1, f"Command should exit with error code 1, got {result.exit_code}"
        assert "Invalid target name:" in result.output, f"Should show invalid name error, got: {result.output}"
        assert (
            expected_error_message in result.output
        ), f"Should include specific error details '{expected_error_message}', got: {result.output}"

    @pytest.mark.unit
    def test_target_already_exists_error(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test target command exits with error when target already exists.

        This unit test verifies proper error handling when FileExistsError
        is raised during target creation, ensuring appropriate error messages.
        """
        # Arrange
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        target_name = "existing_target"

        mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_target",
            side_effect=FileExistsError("Target already exists"),
        )
        mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
        mocker.patch(
            "marimba.core.wrappers.target.DistributionTargetWrapper.prompt_target",
            return_value=("target_type", "target_config"),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "target", target_name, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 1, f"Command should exit with error code 1, got {result.exit_code}"
        assert (
            "A Marimba target already exists at:" in result.output
        ), f"Should show target exists error, got: {result.output}"

    @pytest.mark.unit
    def test_target_creation_failure(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test target command exits with error when target creation fails.

        This unit test verifies proper error handling for general exceptions
        during target creation, ensuring the CLI displays appropriate error messages.
        """
        # Arrange
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        target_name = "test_target"
        expected_error_message = "Creation failed"

        mock_create_target = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_target",
            side_effect=Exception(expected_error_message),
        )
        mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
        mocker.patch(
            "marimba.core.wrappers.target.DistributionTargetWrapper.prompt_target",
            return_value=("target_type", "target_config"),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "target", target_name, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 1, f"Command should exit with error code 1, got {result.exit_code}"
        assert "Could not create target:" in result.output, f"Should show creation error message, got: {result.output}"
        assert (
            expected_error_message in result.output
        ), f"Should include specific error details '{expected_error_message}', got: {result.output}"

        # Verify the correct method was called with expected arguments
        mock_create_target.assert_called_once_with(target_name, "target_type", "target_config")

    @pytest.mark.unit
    def test_target_logs_command_execution(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test target command logs command execution properly.

        This unit test verifies that the CLI command properly logs execution
        information when creating a new distribution target, focusing on testing
        the logging functionality in isolation with mocked dependencies.
        """
        # Arrange
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        target_name = "test_target"

        # Mock external dependencies and project directory detection
        mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)

        mock_target_wrapper = mocker.MagicMock()
        mock_target_wrapper.config_path = project_dir / "targets" / f"{target_name}.yml"

        # Mock target creation to focus on logging behavior
        mock_create_target = mocker.patch(
            "marimba.core.wrappers.project.ProjectWrapper.create_target",
            return_value=mock_target_wrapper,
        )
        mocker.patch(
            "marimba.core.wrappers.target.DistributionTargetWrapper.prompt_target",
            return_value=("target_type", "target_config"),
        )

        # Mock logger to capture logging calls
        mock_logger = mocker.patch("marimba.core.cli.new.logger")

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "target", target_name, "--project-dir", str(project_dir)],
        )

        # Assert
        assert result.exit_code == 0, f"Command should succeed, got output: {result.output}"
        assert "Created new Marimba target" in result.output, "Should show success message"
        assert f'"{target_name}"' in result.output, "Should include target name in output"

        # Verify logging behavior - should log command execution with specific message
        mock_logger.info.assert_called_once()
        log_call_args = mock_logger.info.call_args[0][0]
        assert "Executing the" in log_call_args, "Should log command execution start"
        assert "new target" in log_call_args, "Should include specific command name in log message"

        # Verify the create_target method was called with expected arguments
        mock_create_target.assert_called_once_with(target_name, "target_type", "target_config")


# ---------------------------------------------------------------------------------------------------------------------#
# Testing project directory detection from subdirectories
# ---------------------------------------------------------------------------------------------------------------------#


class TestProjectDirectoryDetection:
    """Test class for project directory detection functionality."""

    @pytest.mark.integration
    def test_find_project_dir_or_exit_with_marimba_as_file(self, tmp_path: Path) -> None:
        """
        Test CLI commands properly handle case where .marimba exists as file instead of directory.

        This integration test verifies that when .marimba exists as a file instead of a directory,
        find_project_dir_or_exit cannot locate a valid Marimba project directory and the CLI
        command exits with appropriate error code and message.
        """
        # Arrange
        project_dir = tmp_path / "invalid_project"
        project_dir.mkdir()

        # Create .marimba as a file instead of directory (invalid project structure)
        marimba_file = project_dir / ".marimba"
        marimba_file.write_text("invalid")

        collection_name = "test_collection"

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "collection", collection_name, "--project-dir", str(project_dir)],
        )

        # Assert
        assert (
            result.exit_code == 1
        ), f"Command should exit with error code 1 when .marimba is file, got {result.exit_code}"
        assert "Could not find a" in result.output, f"Should show project not found error, got: {result.output}"
        assert "project" in result.output, f"Should show project not found error, got: {result.output}"

    @pytest.mark.integration
    def test_find_project_dir_or_exit_with_no_read_access(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """
        Test CLI commands properly handle project directory with no read access.

        This integration test verifies that when find_project_dir_or_exit encounters
        no read access permissions during project directory search, the CLI command
        exits with appropriate error code and message, testing the real interaction
        between file system permission handling and CLI error reporting.
        """
        # Arrange
        collection_name = "test_collection"

        # Mock find_project_dir within the paths module to return None due to no read access
        # This simulates the scenario where find_project_dir cannot access directories
        mock_find_project_dir = mocker.patch("marimba.core.utils.paths.find_project_dir")
        mock_find_project_dir.return_value = None

        # Act - Don't specify project-dir to avoid Typer's path validation,
        # letting find_project_dir_or_exit handle the search from current directory
        result = runner.invoke(
            marimba_cli,
            ["new", "collection", collection_name],
        )

        # Assert
        assert (
            result.exit_code == 1
        ), f"Command should exit with error code 1 when no read access, got {result.exit_code}"
        assert "Could not find a" in result.output, f"Should show project not found error, got: {result.output}"
        assert "project" in result.output, f"Should show project not found error, got: {result.output}"

    @pytest.mark.integration
    def test_find_project_dir_or_exit_with_invalid_project_dir(self, tmp_path: Path) -> None:
        """
        Test CLI commands properly handle invalid project directory detection.

        This integration test verifies that when find_project_dir_or_exit cannot locate
        a valid Marimba project directory, the CLI command exits with appropriate error
        code and message, testing the real interaction between project detection and CLI error handling.
        """
        # Arrange
        invalid_project_dir = tmp_path / "not_a_project"
        invalid_project_dir.mkdir()  # Create directory but no .marimba subdirectory
        collection_name = "test_collection"

        # Act
        result = runner.invoke(
            marimba_cli,
            ["new", "collection", collection_name, "--project-dir", str(invalid_project_dir)],
        )

        # Assert
        assert (
            result.exit_code == 1
        ), f"Command should exit with error code 1 when project not found, got {result.exit_code}"
        assert "Could not find a" in result.output, f"Should show project not found error, got: {result.output}"
        assert "project" in result.output, f"Should mention project in error message, got: {result.output}"

    @pytest.mark.integration
    def test_find_project_dir_from_subdir_executes_successfully(self, tmp_path: Path) -> None:
        """
        Test CLI commands can find project directory when executed from subdirectory.

        This integration test verifies that CLI commands properly locate the project
        root directory when invoked from within subdirectories of the project,
        testing the real interaction between CLI argument parsing and project detection.
        """
        # Arrange - Create project structure with real project creation
        project_dir = tmp_path / "test_project"

        # Create a real Marimba project
        ProjectWrapper.create(project_dir)

        # Create nested subdirectories in a different location to avoid interfering with collections
        subdir = project_dir / "workspace" / "analysis" / "deep"
        subdir.mkdir(parents=True)

        collection_name = "test_collection"

        # Act - Execute CLI command from subdirectory without specifying --project-dir
        # This tests that find_project_dir_or_exit correctly locates the project root
        original_cwd = Path.cwd()
        try:
            # Change to subdirectory to test directory detection
            os.chdir(subdir)

            result = runner.invoke(
                marimba_cli,
                ["new", "collection", collection_name],
            )

        finally:
            # Always restore original working directory
            os.chdir(original_cwd)

        # Assert - Command should succeed by finding project directory
        assert result.exit_code == 0, f"Command should succeed when run from subdirectory, got output: {result.output}"
        assert "Created new Marimba collection" in result.output, "Success message should be displayed"
        assert f'"{collection_name}"' in result.output, "Collection name should be in output"

        # Verify collection was actually created in correct location
        collections_dir = project_dir / "collections"
        assert collections_dir.exists(), "Collections directory should exist in project root"

        collection_dir = collections_dir / collection_name
        assert collection_dir.exists(), "Collection directory should be created in project collections folder"

    @pytest.mark.integration
    def test_find_project_dir_or_exit_with_symlink(self, tmp_path: Path) -> None:
        """
        Test CLI commands properly handle project directory accessed through symlink.

        This integration test verifies that when a Marimba project is accessed through
        a symbolic link, find_project_dir_or_exit correctly resolves the project
        directory and CLI commands execute successfully.
        """
        # Arrange
        # Create actual project directory
        real_project_dir = tmp_path / "real_project"
        ProjectWrapper.create(real_project_dir)

        # Create symbolic link to the project
        symlink_project_dir = tmp_path / "symlinked_project"
        symlink_project_dir.symlink_to(real_project_dir)

        collection_name = "test_collection"

        # Act - Use symlink path as project directory
        result = runner.invoke(
            marimba_cli,
            ["new", "collection", collection_name, "--project-dir", str(symlink_project_dir)],
        )

        # Assert
        assert result.exit_code == 0, f"Command should succeed with symlinked project dir, got output: {result.output}"
        assert "Created new Marimba collection" in result.output, "Success message should be displayed"
        assert f'"{collection_name}"' in result.output, "Collection name should be in output"

        # Verify collection was created in the real project directory
        # (The symlink should be resolved to the actual project)
        real_collections_dir = real_project_dir / "collections"
        assert real_collections_dir.exists(), "Collections directory should exist in real project"

        real_collection_dir = real_collections_dir / collection_name
        assert real_collection_dir.exists(), "Collection should be created in real project directory"

        # Also verify accessible through symlink
        symlink_collections_dir = symlink_project_dir / "collections"
        symlink_collection_dir = symlink_collections_dir / collection_name
        assert symlink_collection_dir.exists(), "Collection should be accessible through symlink"
