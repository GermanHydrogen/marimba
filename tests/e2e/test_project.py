"""
End-to-End tests for project lifecycle operations.

These tests validate project creation, deletion, and management workflows.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from marimba.main import marimba_cli as app


@pytest.mark.e2e
class TestProjectLifecycle:
    """Test basic project creation and management."""

    def test_new_project_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test creating a new project and verifying structure."""
        # Test: marimba new project <name>
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])

        assert result.exit_code == 0, f"Failed to create project: {result.stdout}"

        # Verify project structure was created
        assert temp_project_dir.exists()
        assert (temp_project_dir / ".marimba").is_dir()
        assert (temp_project_dir / "pipelines").is_dir()
        assert (temp_project_dir / "collections").is_dir()
        assert (temp_project_dir / "datasets").is_dir()
        assert (temp_project_dir / "targets").is_dir()

    def test_error_handling_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test that project commands handle errors gracefully."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Test operations on non-existent project
        nonexistent_project = temp_project_dir.parent / "nonexistent_project"
        result = runner.invoke(app, ["delete", "collection", "any", "--project-dir", str(nonexistent_project)])
        assert result.exit_code != 0  # Should fail gracefully
