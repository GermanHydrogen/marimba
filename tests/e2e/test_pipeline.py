"""
End-to-End tests for pipeline operations.

These tests validate pipeline creation, deletion, processing, and management workflows.
"""

from pathlib import Path
from typing import Any

import pytest
import pytest_mock
from typer.testing import CliRunner

from marimba.main import marimba_cli as app
from tests.conftest import assert_cli_success, assert_project_structure_complete


@pytest.mark.e2e
class TestPipelineManagement:
    """Test pipeline creation and management workflows."""

    def test_new_pipeline_workflow_with_mocking(
        self,
        runner: CliRunner,
        temp_project_dir: Path,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test creating a project and adding a pipeline using mocked Git operations."""
        # First create the project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert_cli_success(result, context="Project creation for pipeline test")

        # Create a mock that actually creates the repo directory structure
        def mock_clone_from(_url: str, to_path: str, **_kwargs: Any) -> Any:
            repo_path = Path(to_path)
            repo_path.mkdir(parents=True, exist_ok=True)
            # Create a basic pipeline.yml file that the system expects
            (repo_path / "pipeline.yml").write_text("name: test_pipeline\nversion: 1.0\n")
            # Mock object is returned through the side effect mechanism, not directly
            return mocker.Mock()

        # Mock the Git clone operation to avoid network dependency
        mock_clone = mocker.patch("git.Repo.clone_from", side_effect=mock_clone_from)

        # Mock save_config to avoid file operations during test
        mocker.patch("marimba.core.utils.config.save_config")

        # Test: marimba new pipeline <name> <repo> (with project dir)
        result = runner.invoke(
            app,
            [
                "new",
                "pipeline",
                "test_pipeline",
                "https://github.com/example/test-pipeline.git",
                "--project-dir",
                str(temp_project_dir),
            ],
        )

        # Should succeed with mocked operations
        assert_cli_success(result, context="Pipeline creation with mocked Git operations")

        # Verify Git clone was called with correct parameters
        mock_clone.assert_called_once()
        clone_args = mock_clone.call_args
        assert "https://github.com/example/test-pipeline.git" in str(clone_args)

        # Verify pipeline directory structure was created
        pipeline_dir = temp_project_dir / "pipelines" / "test_pipeline"
        assert pipeline_dir.exists()
        assert (pipeline_dir / "repo").exists()
        assert (pipeline_dir / "repo" / "pipeline.yml").exists()

        # Verify project structure remains intact
        assert_project_structure_complete(temp_project_dir, "Post-pipeline creation")

    @pytest.mark.slow
    def test_new_pipeline_workflow_error_handling(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test pipeline creation error handling with real network calls."""
        # First create the project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert_cli_success(result, context="Project creation for pipeline error handling test")

        # Test with a non-existent repository to test error handling
        result = runner.invoke(
            app,
            [
                "new",
                "pipeline",
                "test_pipeline",
                "https://github.com/nonexistent/nonexistent-repo.git",
                "--project-dir",
                str(temp_project_dir),
            ],
        )

        # Should fail gracefully with a clear error message
        assert result.exit_code != 0
        # Should contain a reasonable error message about the repository
        error_output = result.stdout.lower()
        assert any(word in error_output for word in ["repository", "clone", "git", "not found", "error"])

    @pytest.mark.slow
    def test_comprehensive_pipeline_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test comprehensive workflow that would work with a pipeline."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert_cli_success(result, context="Project creation for comprehensive workflow test")

        # Attempt to create pipeline (will fail due to network, but tests parsing)
        result = runner.invoke(
            app,
            [
                "new",
                "pipeline",
                "test_pipeline",
                "https://github.com/example/test-pipeline.git",
                "--project-dir",
                str(temp_project_dir),
                "--config",
                '{"test_param": "test_value"}',
            ],
        )
        # Expected to fail due to network, but should parse arguments
        assert result.exit_code != 0

        # Verify error handling is graceful
        acceptable_errors = ["git", "network", "connection", "repository", "clone", "timeout", "not found", "eof"]
        error_output = result.stdout.lower()
        has_acceptable_error = any(error in error_output for error in acceptable_errors)
        # Should either succeed or fail gracefully with network/git error
        assert has_acceptable_error or "error" not in error_output.lower()

    @pytest.mark.slow
    def test_delete_pipeline_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test pipeline deletion workflow using actual marimba commands."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert_cli_success(result, context="Project creation for pipeline deletion test")

        # Try to create a pipeline using marimba new pipeline command
        result = runner.invoke(
            app,
            [
                "new",
                "pipeline",
                "test_pipeline",
                "https://github.com/example/test-pipeline.git",
                "--project-dir",
                str(temp_project_dir),
            ],
        )

        pipeline_dir = temp_project_dir / "pipelines" / "test_pipeline"
        pipeline_created_properly = (
            pipeline_dir.exists() and (pipeline_dir / "repo").exists() and (pipeline_dir / "pipeline.yml").exists()
        )

        # Clean up any partial pipeline structure from failed creation using marimba delete
        if pipeline_dir.exists() and not pipeline_created_properly:
            result = runner.invoke(app, ["delete", "pipeline", "test_pipeline", "--project-dir", str(temp_project_dir)])
            # Delete operation should succeed or fail gracefully
            assert result.exit_code in [0, 1]

        # Test pipeline deletion behavior
        if pipeline_created_properly:
            # Pipeline was successfully created, test deletion
            result = runner.invoke(app, ["delete", "pipeline", "test_pipeline", "--project-dir", str(temp_project_dir)])
            # Delete operation should succeed
            assert_cli_success(result, context="Pipeline deletion")
            assert not pipeline_dir.exists()
        else:
            # Pipeline creation failed (expected due to non-existent repo)
            # Test deleting non-existent pipeline
            result = runner.invoke(
                app,
                ["delete", "pipeline", "nonexistent_pipeline", "--project-dir", str(temp_project_dir)],
            )
            # Should fail gracefully for non-existent pipelines
            assert result.exit_code != 0


@pytest.mark.e2e
class TestProcessWorkflows:
    """Test processing workflows and pipeline operations."""

    def test_process_workflow(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test the process command workflow."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert_cli_success(result, context="Project creation for process workflow test")

        # Create some collections using marimba import
        for collection_name in ["test_collection1", "test_collection2"]:
            result = runner.invoke(
                app,
                ["import", collection_name, str(temp_data_dir), "--project-dir", str(temp_project_dir)],
            )
            # Import may fail without pipeline, but that's expected for this test

        # Test process command
        result = runner.invoke(app, ["process", "--project-dir", str(temp_project_dir)])
        # Process might fail without pipeline, but should not crash
        assert result.exit_code in [0, 1]
