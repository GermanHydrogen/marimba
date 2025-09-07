"""Tests for marimba.core.cli.delete module."""

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from marimba.core.cli.delete import (
    batch_delete_operation,
    print_results,
)
from marimba.core.wrappers.project import ProjectWrapper
from marimba.main import marimba_cli
from tests.conftest import assert_cli_failure, assert_cli_success

runner = CliRunner()


@pytest.fixture
def setup_project_dir(tmp_path: Path) -> Path:
    """Set up a test project directory."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    (project_dir / ".marimba").mkdir()
    return project_dir


@pytest.mark.unit
def test_batch_delete_operation_success():
    """Test batch_delete_operation with successful operations."""
    items = ["item1", "item2"]

    def mock_delete_func(name: str, _dry_run: bool) -> Path:
        return Path(f"/path/to/{name}")

    success_items, errors = batch_delete_operation(items, mock_delete_func, "test_entity", "Testing...", False)

    assert len(success_items) == 2
    assert len(errors) == 0
    assert success_items[0] == ("item1", Path("/path/to/item1"))
    assert success_items[1] == ("item2", Path("/path/to/item2"))


@pytest.mark.unit
def test_batch_delete_operation_with_errors():
    """Test batch_delete_operation with some errors."""
    items = ["item1", "bad_item", "item3"]

    def mock_delete_func(name: str, _dry_run: bool) -> Path:
        if name == "bad_item":
            msg = "Collection not found"
            raise ProjectWrapper.NoSuchCollectionError(msg)
        return Path(f"/path/to/{name}")

    success_items, errors = batch_delete_operation(items, mock_delete_func, "collection", "Deleting...", False)

    assert len(success_items) == 2
    assert len(errors) == 1
    assert errors[0] == ("bad_item", "Collection not found")
    assert ("item1", Path("/path/to/item1")) in success_items
    assert ("item3", Path("/path/to/item3")) in success_items


@pytest.mark.unit
def test_batch_delete_operation_handles_all_exceptions():
    """Test that batch_delete_operation handles all expected exception types."""
    exception_types = [
        ProjectWrapper.NoSuchCollectionError("No collection"),
        ProjectWrapper.NoSuchPipelineError("No pipeline"),
        ProjectWrapper.NoSuchDatasetError("No dataset"),
        ProjectWrapper.NoSuchTargetError("No target"),
        ProjectWrapper.DeletePipelineError("Delete error"),
        ProjectWrapper.InvalidNameError("Invalid name"),
        FileExistsError("File exists"),
        Exception("Unexpected error"),
    ]

    for i, exception in enumerate(exception_types):
        items = [f"item_{i}"]

        def mock_delete_func(_name: str, _dry_run: bool, exc: Exception = exception) -> Path:
            raise exc

        success_items, errors = batch_delete_operation(items, mock_delete_func, "entity", "Testing...", False)

        assert len(success_items) == 0
        assert len(errors) == 1
        assert errors[0][0] == f"item_{i}"


@pytest.mark.unit
def test_print_results_success_only():
    """Test print_results with only successful operations."""
    success_items = [
        ("item1", Path("/path/to/item1")),
        ("item2", Path("/path/to/item2")),
    ]
    errors: list[tuple[str, str]] = []

    # Should not raise an exception
    print_results(success_items, errors, "entity")


@pytest.mark.unit
def test_print_results_with_errors():
    """Test print_results with errors raises typer.Exit."""
    success_items = [("item1", Path("/path/to/item1"))]
    errors = [("bad_item", "Error message")]

    with pytest.raises(typer.Exit) as exc_info:
        print_results(success_items, errors, "entity")

    assert exc_info.value.exit_code == 1


@pytest.mark.integration
def test_delete_project_command(mocker, setup_project_dir):
    """Test delete project command."""
    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mock_delete = mocker.patch.object(ProjectWrapper, "delete_project", return_value=setup_project_dir)
    result = runner.invoke(marimba_cli, ["delete", "project", "--project-dir", str(setup_project_dir)])

    assert_cli_success(result, expected_message="Deleted", context="Project deletion")
    mock_delete.assert_called_once()


@pytest.mark.integration
def test_delete_project_invalid_structure(mocker, setup_project_dir):
    """Test delete project with invalid project structure."""
    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mocker.patch.object(ProjectWrapper, "__init__", side_effect=ProjectWrapper.InvalidStructureError("Invalid"))
    result = runner.invoke(marimba_cli, ["delete", "project", "--project-dir", str(setup_project_dir)])

    assert_cli_failure(
        result,
        expected_error="not valid project",
        expected_exit_code=1,
        context="Invalid project structure",
    )


@pytest.mark.integration
def test_delete_project_dry_run(mocker, setup_project_dir):
    """Test delete project with dry run option."""
    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mock_delete = mocker.patch.object(ProjectWrapper, "delete_project", return_value=setup_project_dir)
    result = runner.invoke(marimba_cli, ["delete", "project", "--project-dir", str(setup_project_dir), "--dry-run"])

    assert_cli_success(result, context="Project deletion dry run")
    mock_delete.assert_called_once()


@pytest.mark.integration
def test_delete_pipeline_command(mocker, setup_project_dir):
    """Test delete pipeline command."""
    pipeline_names = ["pipeline1", "pipeline2"]

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mock_delete = mocker.patch.object(ProjectWrapper, "delete_pipeline", return_value=Path("/path"))
    result = runner.invoke(
        marimba_cli,
        ["delete", "pipeline", *pipeline_names, "--project-dir", str(setup_project_dir)],
    )

    assert_cli_success(result, context="Pipeline deletion batch")
    assert mock_delete.call_count == 2
    mock_delete.assert_any_call("pipeline1", False)
    mock_delete.assert_any_call("pipeline2", False)


@pytest.mark.integration
def test_delete_pipeline_no_such_pipeline(mocker, setup_project_dir):
    """Test delete pipeline with non-existent pipeline."""
    pipeline_names = ["nonexistent"]

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mocker.patch.object(
        ProjectWrapper,
        "delete_pipeline",
        side_effect=ProjectWrapper.NoSuchPipelineError("Pipeline not found"),
    )
    result = runner.invoke(
        marimba_cli,
        ["delete", "pipeline", *pipeline_names, "--project-dir", str(setup_project_dir)],
    )

    assert_cli_failure(
        result,
        expected_error="Failed to delete",
        expected_exit_code=1,
        context="Pipeline deletion with missing pipeline",
    )


@pytest.mark.integration
def test_delete_collection_command(mocker, setup_project_dir):
    """Test delete collection command."""
    collection_names = ["collection1", "collection2"]

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mock_delete = mocker.patch.object(ProjectWrapper, "delete_collection", return_value=Path("/path"))
    result = runner.invoke(
        marimba_cli,
        ["delete", "collection", *collection_names, "--project-dir", str(setup_project_dir)],
    )

    assert_cli_success(result, context="Collection deletion batch")
    assert mock_delete.call_count == 2
    mock_delete.assert_any_call("collection1", False)
    mock_delete.assert_any_call("collection2", False)


@pytest.mark.integration
def test_delete_collection_no_such_collection(mocker, setup_project_dir):
    """Test delete collection with non-existent collection."""
    collection_names = ["nonexistent"]

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mocker.patch.object(
        ProjectWrapper,
        "delete_collection",
        side_effect=ProjectWrapper.NoSuchCollectionError("Collection not found"),
    )
    result = runner.invoke(
        marimba_cli,
        ["delete", "collection", *collection_names, "--project-dir", str(setup_project_dir)],
    )

    assert_cli_failure(result, expected_exit_code=1, context="Collection deletion with missing collection")
    assert "Failed to delete" in result.output


@pytest.mark.integration
def test_delete_target_command(mocker, setup_project_dir):
    """Test delete target command."""
    target_names = ["target1", "target2"]

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mock_delete = mocker.patch.object(ProjectWrapper, "delete_target", return_value=Path("/path"))
    result = runner.invoke(marimba_cli, ["delete", "target", *target_names, "--project-dir", str(setup_project_dir)])

    assert_cli_success(result, context="Target deletion batch")
    assert mock_delete.call_count == 2
    mock_delete.assert_any_call("target1", False)
    mock_delete.assert_any_call("target2", False)


@pytest.mark.integration
def test_delete_target_no_such_target(mocker, setup_project_dir):
    """Test delete target with non-existent target."""
    target_names = ["nonexistent"]

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mocker.patch.object(
        ProjectWrapper,
        "delete_target",
        side_effect=ProjectWrapper.NoSuchTargetError("Target not found"),
    )
    result = runner.invoke(marimba_cli, ["delete", "target", *target_names, "--project-dir", str(setup_project_dir)])

    assert_cli_failure(
        result,
        expected_error="Failed to delete",
        expected_exit_code=1,
        context="Target deletion with missing target",
    )


@pytest.mark.integration
def test_delete_dataset_command(mocker, setup_project_dir):
    """Test delete dataset command."""
    dataset_names = ["dataset1", "dataset2"]

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mock_delete = mocker.patch.object(ProjectWrapper, "delete_dataset", return_value=Path("/path"))
    result = runner.invoke(
        marimba_cli,
        ["delete", "dataset", *dataset_names, "--project-dir", str(setup_project_dir)],
    )

    assert_cli_success(result, context="Dataset deletion batch")
    assert mock_delete.call_count == 2
    mock_delete.assert_any_call("dataset1", False)
    mock_delete.assert_any_call("dataset2", False)


@pytest.mark.integration
def test_delete_dataset_no_such_dataset(mocker, setup_project_dir):
    """Test delete dataset with non-existent dataset."""
    dataset_names = ["nonexistent"]

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mocker.patch.object(ProjectWrapper, "delete_dataset", side_effect=FileExistsError("Dataset not found"))
    result = runner.invoke(
        marimba_cli,
        ["delete", "dataset", *dataset_names, "--project-dir", str(setup_project_dir)],
    )

    assert_cli_failure(
        result,
        expected_error="Failed to delete",
        expected_exit_code=1,
        context="Dataset deletion with missing dataset",
    )


@pytest.mark.integration
def test_delete_command_with_dry_run(mocker):
    """Test that dry_run flag is properly passed to delete operations."""
    pipeline_names = ["pipeline1"]

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=Path("/project"))
    mocker.patch.object(ProjectWrapper, "__init__", return_value=None)
    mock_delete = mocker.patch.object(ProjectWrapper, "delete_pipeline", return_value=Path("/path"))
    result = runner.invoke(marimba_cli, ["delete", "pipeline", *pipeline_names, "--dry-run"])

    assert_cli_success(result, context="Pipeline deletion dry run")
    mock_delete.assert_called_once_with("pipeline1", True)


@pytest.mark.integration
def test_delete_multiple_items_mixed_results(mocker, setup_project_dir):
    """Test deleting multiple items with mixed success/failure results."""
    pipeline_names = ["success1", "fail", "success2"]

    def mock_delete_pipeline(name: str, _dry_run: bool) -> Path:
        if name == "fail":
            msg = "Pipeline not found"
            raise ProjectWrapper.NoSuchPipelineError(msg)
        return Path(f"/path/to/{name}")

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mocker.patch.object(ProjectWrapper, "delete_pipeline", side_effect=mock_delete_pipeline)
    result = runner.invoke(
        marimba_cli,
        ["delete", "pipeline", *pipeline_names, "--project-dir", str(setup_project_dir)],
    )

    assert_cli_failure(result, expected_exit_code=1, context="Mixed batch operation with partial failures")
    assert "Deleted" in result.output  # Should show success messages
    assert "Failed to delete" in result.output  # Should show error message
