"""
End-to-End tests for project lifecycle operations.

These tests validate project creation, deletion, and management workflows.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from marimba.main import marimba_cli as app
from tests.conftest import assert_cli_failure, assert_cli_success, assert_project_structure_complete


@pytest.mark.e2e
class TestProjectLifecycle:
    """Test basic project creation and management."""

    def test_new_project_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test creating a new project and verifying structure."""
        # Test: marimba new project <name>
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])

        assert_cli_success(result, context="Project creation workflow")

        # Verify project structure was created
        assert_project_structure_complete(temp_project_dir, "New project creation")

    def test_error_handling_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test that project commands handle errors gracefully."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert_cli_success(result, context="Project creation for error handling test")

        # Test operations on non-existent project
        nonexistent_project = temp_project_dir.parent / "nonexistent_project"
        result = runner.invoke(app, ["delete", "collection", "any", "--project-dir", str(nonexistent_project)])
        assert_cli_failure(result, context="Operation on non-existent project")  # Should fail gracefully
