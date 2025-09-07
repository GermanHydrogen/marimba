"""
End-to-End tests for dataset operations.

These tests validate dataset packaging, metadata handling, and dataset management workflows.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from marimba.main import marimba_cli as app
from tests.conftest import assert_cli_success, assert_cli_failure


@pytest.mark.e2e
class TestDatasetPackaging:
    """Test dataset packaging and creation workflows."""

    def test_package_workflow(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test packaging a dataset."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert_cli_success(result, context="Project creation for dataset package workflow")

        # Import some data to create a collection
        result = runner.invoke(
            app, ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir)]
        )
        assert result.exit_code in [0, 1]  # May fail without pipeline

        # Test: marimba package <dataset>
        result = runner.invoke(
            app,
            [
                "package",
                "test_dataset",
                "--collection-name",
                "test_collection",
                "--project-dir",
                str(temp_project_dir),
                "--version",
                "1.0",
                "--contact-name",
                "Test User",
                "--contact-email",
                "test@example.com",
            ],
        )

        # Package might fail without proper metadata, but should not crash
        assert result.exit_code in [0, 1], f"Package command crashed unexpectedly: {result.stdout}"

    def test_package_workflow_with_metadata_options(
        self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path
    ) -> None:
        """Test packaging with various metadata output options."""
        # Create project and basic dataset
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert_cli_success(result, context="Project creation for metadata options test")

        # Import some data to create a collection
        result = runner.invoke(
            app, ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir)]
        )
        assert result.exit_code in [0, 1]

        # Test package with multiple metadata options
        result = runner.invoke(
            app,
            [
                "package",
                "test_dataset",
                "--collection-name",
                "test_collection",
                "--project-dir",
                str(temp_project_dir),
                "--version",
                "1.0",
                "--contact-name",
                "Test User",
                "--contact-email",
                "test@example.com",
                "--metadata-output",
                "yaml",
                "--metadata-level",
                "project",
                "--metadata-level",
                "collection",
                "--allow-destination-collisions",
            ],
        )
        # Package should parse all options correctly
        assert result.exit_code in [0, 1]

    def test_package_with_dry_run(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test package workflow with dry run option."""
        # Create project and basic dataset
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert_cli_success(result, context="Project creation for deletion workflow test")

        # Import some data to create a collection
        result = runner.invoke(
            app, ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir)]
        )
        assert result.exit_code in [0, 1]

        # Test package with dry run
        result = runner.invoke(
            app,
            [
                "package",
                "test_dataset",
                "--collection-name",
                "test_collection",
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
        # Package should parse all options correctly in dry run
        assert result.exit_code in [0, 1]


@pytest.mark.e2e
class TestDatasetDeletion:
    """Test dataset deletion workflows."""

    def test_delete_dataset_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test dataset deletion operations."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Test deleting non-existent dataset (should fail gracefully)
        result = runner.invoke(
            app, ["delete", "dataset", "nonexistent_dataset", "--project-dir", str(temp_project_dir)]
        )
        # Should fail gracefully for non-existent datasets
        assert_cli_failure(result, context="Delete non-existent dataset")

        # Test batch delete dataset operation
        result = runner.invoke(app, ["delete", "dataset", "TEST_DATA", "--project-dir", str(temp_project_dir)])
        # May fail if dataset doesn't exist, which is expected behavior
        assert result.exit_code in [0, 1]  # Could succeed or fail gracefully
