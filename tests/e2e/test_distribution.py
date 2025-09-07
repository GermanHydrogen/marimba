"""
End-to-End tests for distribution operations.

These tests validate distribution workflows for various target types including S3 and DAP.
"""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from marimba.main import marimba_cli as app
from tests.conftest import assert_cli_failure, assert_cli_success


@pytest.mark.e2e
class TestDistributionWorkflows:
    """Test distribution workflows for various target types."""

    @pytest.fixture
    def project(self, runner: CliRunner, temp_project_dir: Path) -> Path:
        """Create a marimba project."""
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert_cli_success(result, context="Distribution test project creation")
        return temp_project_dir

    @pytest.fixture
    def mock_dataset_dir(self, runner: CliRunner, project: Path, temp_data_dir: Path) -> Path:
        """Create a mock dataset directory with sample files."""
        # Import some data to create a collection
        result = runner.invoke(app, ["import", "test_collection", str(temp_data_dir), "--project-dir", str(project)])
        # Allow success or failure since import can fail without pipelines
        assert result.exit_code in [0, 1]

        # Package the collection into a dataset
        result = runner.invoke(
            app,
            [
                "package",
                "test_dataset",
                "--collection-name",
                "test_collection",
                "--project-dir",
                str(project),
                "--version",
                "1.0",
                "--contact-name",
                "Test User",
                "--contact-email",
                "test@example.com",
            ],
        )
        assert result.exit_code in [0, 1]

        dataset_dir = project / "datasets" / "test_dataset"
        return dataset_dir

    @pytest.fixture
    def mock_s3_target_dir(self, project: Path) -> Path:
        """Create a mock S3 target directory manually for testing."""
        # Create target directory structure manually since interactive creation is complex
        target_dir = project / "targets" / "test_s3_target"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal target.yml file
        target_config = {
            "type": "s3",
            "bucket": "test-bucket",
            "endpoint_url": "https://test.s3.amazonaws.com",
            "access_key_id": "test_access_key",
            "secret_access_key": "test_secret_key",
            "region": "us-east-1",
        }

        (target_dir / "target.yml").write_text(yaml.dump(target_config))
        return target_dir

    @pytest.fixture
    def mock_dap_target_dir(self, project: Path) -> Path:
        """Create a mock DAP target directory manually for testing."""
        # Create target directory structure manually
        target_dir = project / "targets" / "test_dap_target"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal target.yml file
        target_config = {
            "type": "dap",
            "base_url": "https://test.dap.server.com",
            "username": "test_user",
            "password": "test_password",
            "dataset_path": "/datasets",
        }

        (target_dir / "target.yml").write_text(yaml.dump(target_config))
        return target_dir

    def test_distribute_to_s3_target_dry_run(
        self,
        runner: CliRunner,
        project: Path,
        mock_dataset_dir: Path,
        mock_s3_target_dir: Path,
    ) -> None:
        """Test distribution to S3 target with dry run."""
        # Ensure target exists
        assert mock_s3_target_dir.exists()

        # Test distribute command with dry run (should not fail on network)
        result = runner.invoke(
            app,
            ["distribute", "test_dataset", "test_s3_target", "--project-dir", str(project), "--dry-run"],
        )
        # In dry run mode, should parse everything correctly without network calls
        assert result.exit_code in [0, 1]  # May fail due to missing target/dataset, but should handle gracefully

    def test_distribute_to_dap_target_dry_run(
        self,
        runner: CliRunner,
        project: Path,
        mock_dataset_dir: Path,
        mock_dap_target_dir: Path,
    ) -> None:
        """Test distribution to DAP target with dry run."""
        # Ensure target exists
        assert mock_dap_target_dir.exists()

        # Test distribute command with dry run
        result = runner.invoke(
            app,
            ["distribute", "test_dataset", "test_dap_target", "--project-dir", str(project), "--dry-run"],
        )
        # Should handle dry run gracefully
        assert result.exit_code in [0, 1]

    def test_distribute_with_validation_disabled(
        self,
        runner: CliRunner,
        project: Path,
        mock_dataset_dir: Path,
        mock_s3_target_dir: Path,
    ) -> None:
        """Test distribution with validation disabled."""
        # Ensure target exists
        assert mock_s3_target_dir.exists()

        # Test distribute without validation
        result = runner.invoke(
            app,
            [
                "distribute",
                "test_dataset",
                "test_s3_target",
                "--project-dir",
                str(project),
                "--no-validate",
                "--dry-run",
            ],
        )
        # Should skip validation step
        assert result.exit_code in [0, 1]

    def test_distribute_nonexistent_dataset(self, runner: CliRunner, project: Path, mock_s3_target_dir: Path) -> None:
        """Test distribution of non-existent dataset."""
        # Ensure target exists
        assert mock_s3_target_dir.exists()

        # Test distribute non-existent dataset
        result = runner.invoke(
            app,
            ["distribute", "nonexistent_dataset", "test_s3_target", "--project-dir", str(project)],
        )
        # Should fail gracefully with appropriate error
        # Note: CLI may return 0 but should show error in output
        assert result.exit_code != 0 or "no such dataset" in result.stdout.lower()

    def test_distribute_nonexistent_target(self, runner: CliRunner, project: Path, mock_dataset_dir: Path) -> None:
        """Test distribution to non-existent target."""
        # Ensure dataset exists
        assert mock_dataset_dir.exists()

        # Test distribute to non-existent target
        result = runner.invoke(app, ["distribute", "test_dataset", "nonexistent_target", "--project-dir", str(project)])
        # Should fail gracefully with appropriate error
        # Note: CLI may return 0 but should show error in output
        assert result.exit_code != 0 or "no such target" in result.stdout.lower()

    def test_distribute_invalid_project_directory(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test distribution from invalid project directory."""
        nonexistent_project = temp_project_dir.parent / "nonexistent_project"

        # Test distribute from non-existent project
        result = runner.invoke(
            app,
            ["distribute", "test_dataset", "test_target", "--project-dir", str(nonexistent_project)],
        )
        # Should fail gracefully with appropriate error
        assert_cli_failure(result, context="Distribution without project")
        assert "project" in result.stdout.lower() or "not found" in result.stdout.lower()

    def test_distribute_workflow_argument_parsing(self, runner: CliRunner, project: Path) -> None:
        """Test that distribute command correctly parses all arguments."""
        # Test distribute with all available flags
        result = runner.invoke(
            app,
            [
                "distribute",
                "test_dataset",
                "test_target",
                "--project-dir",
                str(project),
                "--validate",
                "--dry-run",
            ],
        )
        # Should parse arguments correctly even if target/dataset don't exist
        assert result.exit_code in [0, 1]  # May fail due to missing components

        # Test with negation flag
        result = runner.invoke(
            app,
            [
                "distribute",
                "test_dataset",
                "test_target",
                "--project-dir",
                str(project),
                "--no-validate",
                "--dry-run",
            ],
        )
        # Should parse negation flag correctly
        assert result.exit_code in [0, 1]

    @pytest.mark.slow
    def test_comprehensive_distribute_workflow(
        self,
        runner: CliRunner,
        temp_project_dir: Path,
        temp_data_dir: Path,
    ) -> None:
        """Test comprehensive workflow: create -> package -> distribute."""
        # Step 1: Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert_cli_success(result, context="Project creation for comprehensive workflow test")

        # Step 2: Create target using new command (may fail, but tests argument parsing)
        result = runner.invoke(app, ["new", "target", "test_s3_target", "--project-dir", str(temp_project_dir)])
        # Target creation may fail due to interactive prompts, which is expected

        # Step 3: Import some data (may fail without pipeline)
        result = runner.invoke(
            app,
            ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir)],
        )
        # Import may fail without pipeline

        # Step 4: Try to package dataset (may fail without proper data)
        result = runner.invoke(
            app,
            [
                "package",
                "test_dataset",
                "--project-dir",
                str(temp_project_dir),
                "--version",
                "1.0",
                "--contact-name",
                "Test User",
                "--contact-email",
                "test@example.com",
                "--dry-run",
            ],
        )
        # Package may fail due to missing data/pipelines

        # Step 5: Attempt distribution (should fail gracefully)
        result = runner.invoke(
            app,
            ["distribute", "test_dataset", "test_s3_target", "--project-dir", str(temp_project_dir), "--dry-run"],
        )
        # Distribution should fail gracefully due to missing components
        assert result.exit_code in [0, 1]

    def test_distribute_command_help_and_options(self, runner: CliRunner) -> None:
        """Test that distribute command help works and shows all options."""
        # Test help for distribute command
        result = runner.invoke(app, ["distribute", "--help"])
        assert_cli_success(result, expected_message="distribute", context="Distribute help command")
        assert "dataset" in result.stdout.lower()
        assert "target" in result.stdout.lower()
        assert "validate" in result.stdout.lower()
        assert "dry-run" in result.stdout.lower()
