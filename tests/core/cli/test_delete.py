"""Tests for marimba.core.cli.delete module."""

from pathlib import Path

import pytest
import pytest_mock
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


class TestBatchDeleteOperationMixedResults:
    """Test batch_delete_operation function with mixed success/error scenarios."""

    @pytest.mark.unit
    def test_batch_delete_operation_handles_partial_failures(self):
        """Test that batch_delete_operation correctly handles mixed success and failure outcomes.

        This test verifies that when some items succeed and others fail during batch deletion,
        the function properly separates successful operations from failed ones and preserves
        the exact error messages from exceptions.
        """
        # Arrange
        items = ["successful_item1", "failing_item", "successful_item2"]
        expected_error_message = "Collection 'failing_item' not found in project"

        def mock_delete_func(name: str, _dry_run: bool) -> Path:
            if name == "failing_item":
                raise ProjectWrapper.NoSuchCollectionError(expected_error_message)
            return Path(f"/mock/path/to/{name}")

        # Act
        success_items, errors = batch_delete_operation(
            items,
            mock_delete_func,
            "collection",
            "Deleting collections...",
            False,
        )

        # Assert
        # Verify counts
        assert len(success_items) == 2, "Should have exactly 2 successful operations"
        assert len(errors) == 1, "Should have exactly 1 failed operation"

        # Verify error details
        assert errors[0] == (
            "failing_item",
            expected_error_message,
        ), "Error should contain exact item name and error message"

        # Verify successful operations (order preserved)
        expected_success_items = [
            ("successful_item1", Path("/mock/path/to/successful_item1")),
            ("successful_item2", Path("/mock/path/to/successful_item2")),
        ]
        assert (
            success_items == expected_success_items
        ), "Success items should contain exact names and paths in processing order"

    @pytest.mark.unit
    def test_batch_delete_operation_preserves_processing_order(self):
        """Test that batch_delete_operation preserves the order of successful operations.

        This verifies that successful items are returned in the same order they were
        processed, even when there are failures interspersed.
        """
        # Arrange
        items = ["first", "second_fails", "third", "fourth_fails", "fifth"]

        def mock_delete_func(name: str, _dry_run: bool) -> Path:
            if "fails" in name:
                error_msg = f"Failed to delete {name}"
                raise ProjectWrapper.NoSuchCollectionError(error_msg)
            return Path(f"/order/test/{name}")

        # Act
        success_items, errors = batch_delete_operation(
            items,
            mock_delete_func,
            "collection",
            "Testing order...",
            False,
        )

        # Assert
        # Verify successful operations maintain order
        expected_successful_names = ["first", "third", "fifth"]
        actual_successful_names = [name for name, _ in success_items]
        assert (
            actual_successful_names == expected_successful_names
        ), "Successful items should maintain their original processing order"

        # Verify failed operations maintain order
        expected_failed_names = ["second_fails", "fourth_fails"]
        actual_failed_names = [name for name, _ in errors]
        assert (
            actual_failed_names == expected_failed_names
        ), "Failed items should maintain their original processing order"

        # Verify counts
        assert len(success_items) == 3, "Should have 3 successful operations"
        assert len(errors) == 2, "Should have 2 failed operations"


class TestBatchDeleteOperationExceptionHandling:
    """Test exception handling in batch_delete_operation function."""

    @pytest.mark.unit
    def test_handles_no_such_collection_error(self):
        """Test that NoSuchCollectionError is properly handled and error message captured."""
        # Arrange
        items = ["missing_collection"]
        error_message = "Collection 'missing_collection' not found"

        def mock_delete_func(_name: str, _dry_run: bool) -> Path:
            raise ProjectWrapper.NoSuchCollectionError(error_message)

        # Act
        success_items, errors = batch_delete_operation(items, mock_delete_func, "collection", "Testing...", False)

        # Assert
        assert len(success_items) == 0, "Should have no successful deletions"
        assert len(errors) == 1, "Should have exactly one error"
        assert errors[0] == ("missing_collection", error_message), "Should capture exact item name and error message"

    @pytest.mark.unit
    def test_handles_no_such_pipeline_error(self):
        """Test that NoSuchPipelineError is properly handled and error message captured."""
        # Arrange
        items = ["missing_pipeline"]
        error_message = "Pipeline 'missing_pipeline' not found"

        def mock_delete_func(_name: str, _dry_run: bool) -> Path:
            raise ProjectWrapper.NoSuchPipelineError(error_message)

        # Act
        success_items, errors = batch_delete_operation(items, mock_delete_func, "pipeline", "Testing...", False)

        # Assert
        assert len(success_items) == 0, "Should have no successful deletions"
        assert len(errors) == 1, "Should have exactly one error"
        assert errors[0] == ("missing_pipeline", error_message), "Should capture exact item name and error message"

    @pytest.mark.unit
    def test_handles_no_such_dataset_error(self):
        """Test that NoSuchDatasetError is properly handled and error message captured."""
        # Arrange
        items = ["missing_dataset"]
        error_message = "Dataset 'missing_dataset' not found"

        def mock_delete_func(_name: str, _dry_run: bool) -> Path:
            raise ProjectWrapper.NoSuchDatasetError(error_message)

        # Act
        success_items, errors = batch_delete_operation(items, mock_delete_func, "dataset", "Testing...", False)

        # Assert
        assert len(success_items) == 0, "Should have no successful deletions"
        assert len(errors) == 1, "Should have exactly one error"
        assert errors[0] == ("missing_dataset", error_message), "Should capture exact item name and error message"

    @pytest.mark.unit
    def test_handles_no_such_target_error(self):
        """Test that NoSuchTargetError is properly handled and error message captured."""
        # Arrange
        items = ["missing_target"]
        error_message = "Target 'missing_target' not found"

        def mock_delete_func(_name: str, _dry_run: bool) -> Path:
            raise ProjectWrapper.NoSuchTargetError(error_message)

        # Act
        success_items, errors = batch_delete_operation(items, mock_delete_func, "target", "Testing...", False)

        # Assert
        assert len(success_items) == 0, "Should have no successful deletions"
        assert len(errors) == 1, "Should have exactly one error"
        assert errors[0] == ("missing_target", error_message), "Should capture exact item name and error message"

    @pytest.mark.unit
    def test_handles_delete_pipeline_error(self):
        """Test that DeletePipelineError is properly handled and error message captured."""
        # Arrange
        items = ["problematic_pipeline"]
        error_message = "Cannot delete pipeline due to dependency"

        def mock_delete_func(_name: str, _dry_run: bool) -> Path:
            raise ProjectWrapper.DeletePipelineError(error_message)

        # Act
        success_items, errors = batch_delete_operation(items, mock_delete_func, "pipeline", "Testing...", False)

        # Assert
        assert len(success_items) == 0, "Should have no successful deletions"
        assert len(errors) == 1, "Should have exactly one error"
        assert errors[0] == ("problematic_pipeline", error_message), "Should capture exact item name and error message"

    @pytest.mark.unit
    def test_handles_invalid_name_error(self):
        """Test that InvalidNameError is properly handled and error message captured."""
        # Arrange
        items = ["invalid@name"]
        error_message = "Invalid characters in name"

        def mock_delete_func(_name: str, _dry_run: bool) -> Path:
            raise ProjectWrapper.InvalidNameError(error_message)

        # Act
        success_items, errors = batch_delete_operation(items, mock_delete_func, "entity", "Testing...", False)

        # Assert
        assert len(success_items) == 0, "Should have no successful deletions"
        assert len(errors) == 1, "Should have exactly one error"
        assert errors[0] == ("invalid@name", error_message), "Should capture exact item name and error message"

    @pytest.mark.unit
    def test_handles_file_exists_error(self):
        """Test that FileExistsError is properly handled and error message captured."""
        # Arrange
        items = ["nonexistent_dataset"]
        error_message = "Dataset file does not exist"

        def mock_delete_func(_name: str, _dry_run: bool) -> Path:
            raise FileExistsError(error_message)

        # Act
        success_items, errors = batch_delete_operation(items, mock_delete_func, "dataset", "Testing...", False)

        # Assert
        assert len(success_items) == 0, "Should have no successful deletions"
        assert len(errors) == 1, "Should have exactly one error"
        assert errors[0] == ("nonexistent_dataset", error_message), "Should capture exact item name and error message"

    @pytest.mark.unit
    def test_handles_unexpected_exception(self):
        """Test that unexpected exceptions are properly handled and error message captured."""
        # Arrange
        items = ["problematic_item"]
        error_message = "Unexpected system error"

        def mock_delete_func(_name: str, _dry_run: bool) -> Path:
            raise RuntimeError(error_message)

        # Act
        success_items, errors = batch_delete_operation(items, mock_delete_func, "entity", "Testing...", False)

        # Assert
        assert len(success_items) == 0, "Should have no successful deletions"
        assert len(errors) == 1, "Should have exactly one error"
        assert errors[0] == ("problematic_item", error_message), "Should capture exact item name and error message"


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
def test_delete_project_command(
    mocker: pytest_mock.MockerFixture,
    setup_project_dir: Path,
) -> None:
    """Test successful deletion of a Marimba project via CLI command.

    This integration test verifies that the delete project CLI command correctly:
    - Finds the project directory using the provided path
    - Creates a ProjectWrapper instance with correct parameters
    - Calls the delete_project method on the wrapper
    - Displays success message with the deleted project path
    - Exits with code 0 on successful completion
    """
    # Arrange
    expected_deleted_path = setup_project_dir

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mock_delete = mocker.patch.object(ProjectWrapper, "delete_project", return_value=expected_deleted_path)

    # Act
    result = runner.invoke(
        marimba_cli,
        ["delete", "project", "--project-dir", str(setup_project_dir)],
    )

    # Assert
    # Verify CLI execution succeeded
    assert result.exit_code == 0, f"CLI command should succeed, got: {result.output}"

    # Verify delete_project was called
    mock_delete.assert_called_once()

    # Verify CLI output contains success message and project path
    assert "Deleted" in result.output, "Should display success message"
    assert str(expected_deleted_path) in result.output, "Should show the deleted project path"


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
def test_delete_project_dry_run(
    mocker: pytest_mock.MockerFixture,
    setup_project_dir: Path,
) -> None:
    """Test delete project command with dry-run flag passes correct parameter.

    This integration test verifies that the delete project CLI command correctly:
    - Parses the --dry-run flag from command line arguments
    - Passes dry_run=True to the ProjectWrapper constructor
    - Still calls the delete_project method (dry-run behavior is handled within ProjectWrapper)
    - Displays success message as if the operation completed
    - Exits with success code 0
    """
    # Arrange
    expected_deleted_path = setup_project_dir

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)

    # Mock ProjectWrapper class to capture initialization arguments
    mock_project_wrapper_class = mocker.patch("marimba.core.cli.delete.ProjectWrapper")
    mock_project_wrapper_instance = mocker.MagicMock()
    mock_project_wrapper_instance.root_dir = setup_project_dir
    mock_project_wrapper_instance.delete_project.return_value = expected_deleted_path
    mock_project_wrapper_class.return_value = mock_project_wrapper_instance

    # Act
    result = runner.invoke(
        marimba_cli,
        ["delete", "project", "--project-dir", str(setup_project_dir), "--dry-run"],
    )

    # Assert
    # Verify CLI execution succeeded
    assert result.exit_code == 0, f"CLI command should succeed with dry-run, got: {result.output}"

    # Verify ProjectWrapper was initialized with dry_run=True (most important assertion for dry-run test)
    mock_project_wrapper_class.assert_called_once_with(setup_project_dir, dry_run=True)

    # Verify delete_project was called (dry-run logic is handled within the ProjectWrapper)
    mock_project_wrapper_instance.delete_project.assert_called_once()

    # Verify CLI output contains success message
    assert "Deleted" in result.output, "Should display success message even in dry-run mode"
    assert str(expected_deleted_path) in result.output, "Should show the project path"


@pytest.mark.integration
def test_delete_pipeline_handles_nonexistent_pipeline_error(
    mocker: pytest_mock.MockerFixture,
    setup_project_dir: Path,
) -> None:
    """Test delete pipeline command properly handles NoSuchPipelineError.

    This integration test verifies that when attempting to delete a pipeline
    that doesn't exist, the CLI command:
    - Catches the NoSuchPipelineError from ProjectWrapper
    - Displays appropriate error message with pipeline name
    - Exits with error code 1
    - Shows the specific error message from the exception
    """
    # Arrange
    nonexistent_pipeline = "missing_analysis_pipeline"
    expected_error_message = f"Pipeline '{nonexistent_pipeline}' not found in project"

    mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
    mock_delete_pipeline = mocker.patch.object(
        ProjectWrapper,
        "delete_pipeline",
        side_effect=ProjectWrapper.NoSuchPipelineError(expected_error_message),
    )

    # Act
    result = runner.invoke(
        marimba_cli,
        ["delete", "pipeline", nonexistent_pipeline, "--project-dir", str(setup_project_dir)],
    )

    # Assert
    # Verify CLI execution failed with correct exit code
    assert result.exit_code == 1, f"CLI should exit with code 1 for missing pipeline, got: {result.output}"

    # Verify ProjectWrapper.delete_pipeline was called
    mock_delete_pipeline.assert_called_once_with(nonexistent_pipeline, False)

    # Verify CLI output contains error messages
    assert "Failed to delete" in result.output, "Should display failure message"
    assert nonexistent_pipeline in result.output, "Should mention the pipeline name that failed"
    assert "not found in project" in result.output, "Should display the specific error message from exception"


class TestDeleteCollectionCommand:
    """Test CLI delete collection command integration."""

    @pytest.mark.integration
    def test_delete_multiple_collections_successful(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test successful deletion of multiple collections via CLI command.

        This integration test verifies that the delete collection CLI command correctly:
        - Processes multiple collection names from command line arguments
        - Calls the ProjectWrapper.delete_collection method for each collection
        - Displays success messages for each deleted collection
        - Exits with code 0 on successful completion
        """
        # Arrange
        collection_names = ["marine_data", "coastal_survey"]
        expected_paths = [
            Path("/project/collections/marine_data"),
            Path("/project/collections/coastal_survey"),
        ]

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_collection = mocker.patch.object(
            ProjectWrapper,
            "delete_collection",
            side_effect=expected_paths,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "collection", *collection_names, "--project-dir", str(setup_project_dir)],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed, got: {result.output}"

        # Verify ProjectWrapper.delete_collection was called correctly for each collection
        assert mock_delete_collection.call_count == 2, "Should call delete_collection twice"
        mock_delete_collection.assert_any_call("marine_data", False)
        mock_delete_collection.assert_any_call("coastal_survey", False)

        # Verify CLI output contains success messages for each collection
        assert "Deleted" in result.output, "Should display success message"
        assert "marine_data" in result.output, "Should mention first collection name"
        assert "coastal_survey" in result.output, "Should mention second collection name"
        assert str(expected_paths[0]) in result.output, "Should show first collection path"
        assert str(expected_paths[1]) in result.output, "Should show second collection path"

    @pytest.mark.integration
    def test_delete_collection_with_dry_run_flag(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete collection command with dry-run flag passes correct parameter.

        This test ensures that the --dry-run flag is properly propagated through
        the CLI to the underlying delete_collection method.
        """
        # Arrange
        collection_names = ["test_collection"]
        expected_path = Path("/project/collections/test_collection")

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_collection = mocker.patch.object(
            ProjectWrapper,
            "delete_collection",
            return_value=expected_path,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "collection", *collection_names, "--project-dir", str(setup_project_dir), "--dry-run"],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed with dry-run, got: {result.output}"

        # Verify delete_collection was called with dry_run=True
        mock_delete_collection.assert_called_once_with("test_collection", True)

        # Verify CLI output contains success message
        assert "Deleted" in result.output, "Should display success message even in dry-run"
        assert "test_collection" in result.output, "Should mention collection name"

    @pytest.mark.integration
    def test_delete_collection_handles_nonexistent_collection_error(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete collection command properly handles NoSuchCollectionError.

        This integration test verifies that when attempting to delete a collection
        that doesn't exist, the CLI command:
        - Catches the NoSuchCollectionError from ProjectWrapper
        - Displays appropriate error message with collection name
        - Exits with error code 1
        - Shows the specific error message from the exception
        """
        # Arrange
        nonexistent_collection = "missing_marine_collection"
        expected_error_message = f"Collection '{nonexistent_collection}' not found in project"

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_collection = mocker.patch.object(
            ProjectWrapper,
            "delete_collection",
            side_effect=ProjectWrapper.NoSuchCollectionError(expected_error_message),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "collection", nonexistent_collection, "--project-dir", str(setup_project_dir)],
        )

        # Assert
        # Verify CLI execution failed with correct exit code
        assert result.exit_code == 1, f"CLI should exit with code 1 for missing collection, got: {result.output}"

        # Verify ProjectWrapper.delete_collection was called
        mock_delete_collection.assert_called_once_with(nonexistent_collection, False)

        # Verify CLI output contains error messages
        assert "Failed to delete" in result.output, "Should display failure message"
        assert nonexistent_collection in result.output, "Should mention the collection name that failed"
        assert "not found in project" in result.output, "Should display the specific error message from exception"

    @pytest.mark.integration
    def test_delete_multiple_collections_with_partial_failures(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete collection command with mixed success and failure results.

        This test verifies that when deleting multiple collections where some exist
        and others don't, the CLI properly:
        - Processes all collections in the batch
        - Shows success messages for existing collections
        - Shows error messages for missing collections
        - Exits with error code 1 due to failures
        """
        # Arrange
        collection_names = ["existing_collection", "missing_collection", "another_existing"]
        existing_path = Path("/project/collections/existing_collection")
        another_existing_path = Path("/project/collections/another_existing")
        missing_error = "Collection 'missing_collection' not found"

        def mock_delete_side_effect(name: str, _dry_run: bool) -> Path:
            if name == "missing_collection":
                raise ProjectWrapper.NoSuchCollectionError(missing_error)
            if name == "existing_collection":
                return existing_path
            if name == "another_existing":
                return another_existing_path
            unexpected_name_msg = f"Unexpected collection name: {name}"
            raise ValueError(unexpected_name_msg)

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_collection = mocker.patch.object(
            ProjectWrapper,
            "delete_collection",
            side_effect=mock_delete_side_effect,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "collection", *collection_names, "--project-dir", str(setup_project_dir)],
        )

        # Assert
        # Verify CLI execution failed due to partial failures
        assert result.exit_code == 1, f"CLI should exit with code 1 for partial failures, got: {result.output}"

        # Verify all collections were attempted
        assert mock_delete_collection.call_count == 3, "Should attempt to delete all 3 collections"
        mock_delete_collection.assert_any_call("existing_collection", False)
        mock_delete_collection.assert_any_call("missing_collection", False)
        mock_delete_collection.assert_any_call("another_existing", False)

        # Verify CLI output contains both success and error messages
        assert "Deleted" in result.output, "Should show success messages for existing collections"
        assert "Failed to delete" in result.output, "Should show failure message for missing collection"
        assert "existing_collection" in result.output, "Should mention successful collection"
        assert "another_existing" in result.output, "Should mention other successful collection"
        assert "missing_collection" in result.output, "Should mention failed collection"
        assert "not found" in result.output, "Should display specific error message"


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


class TestDeleteDatasetCommand:
    """Test CLI delete dataset command integration."""

    @pytest.mark.integration
    def test_delete_multiple_datasets_successful(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test successful deletion of multiple datasets via CLI command.

        This integration test verifies that the delete dataset CLI command correctly:
        - Processes multiple dataset names from command line arguments
        - Calls the ProjectWrapper.delete_dataset method for each dataset
        - Displays success messages for each deleted dataset
        - Exits with code 0 on successful completion
        """
        # Arrange
        dataset_names = ["marine_analysis_2023", "coastal_survey_results"]
        expected_paths = [
            Path("/project/datasets/marine_analysis_2023"),
            Path("/project/datasets/coastal_survey_results"),
        ]

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_dataset = mocker.patch.object(
            ProjectWrapper,
            "delete_dataset",
            side_effect=expected_paths,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "dataset", *dataset_names, "--project-dir", str(setup_project_dir)],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed, got: {result.output}"

        # Verify ProjectWrapper.delete_dataset was called correctly for each dataset
        assert mock_delete_dataset.call_count == 2, "Should call delete_dataset twice"
        mock_delete_dataset.assert_any_call("marine_analysis_2023", False)
        mock_delete_dataset.assert_any_call("coastal_survey_results", False)

        # Verify CLI output contains success messages for each dataset
        assert "Deleted" in result.output, "Should display success message"
        assert "marine_analysis_2023" in result.output, "Should mention first dataset name"
        assert "coastal_survey_results" in result.output, "Should mention second dataset name"
        assert str(expected_paths[0]) in result.output, "Should show first dataset path"
        assert str(expected_paths[1]) in result.output, "Should show second dataset path"

    @pytest.mark.integration
    def test_delete_dataset_with_dry_run_flag(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete dataset command with dry-run flag passes correct parameter.

        This test ensures that the --dry-run flag is properly propagated through
        the CLI to the underlying delete_dataset method.
        """
        # Arrange
        dataset_names = ["test_dataset"]
        expected_path = Path("/project/datasets/test_dataset")

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_dataset = mocker.patch.object(
            ProjectWrapper,
            "delete_dataset",
            return_value=expected_path,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "dataset", *dataset_names, "--project-dir", str(setup_project_dir), "--dry-run"],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed with dry-run, got: {result.output}"

        # Verify delete_dataset was called with dry_run=True
        mock_delete_dataset.assert_called_once_with("test_dataset", True)

        # Verify CLI output contains success message
        assert "Deleted" in result.output, "Should display success message even in dry-run"
        assert "test_dataset" in result.output, "Should mention dataset name"

    @pytest.mark.integration
    def test_delete_dataset_handles_file_exists_error(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete dataset command properly handles FileExistsError for missing datasets.

        This integration test verifies that when attempting to delete a dataset
        that doesn't exist, the CLI command:
        - Catches the FileExistsError from ProjectWrapper (used when dataset doesn't exist)
        - Displays appropriate error message with dataset name
        - Exits with error code 1
        - Shows the specific error message from the exception
        """
        # Arrange
        nonexistent_dataset = "missing_climate_data_2023"
        expected_error_message = f"Dataset '{nonexistent_dataset}' does not exist in project"

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_dataset = mocker.patch.object(
            ProjectWrapper,
            "delete_dataset",
            side_effect=FileExistsError(expected_error_message),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "dataset", nonexistent_dataset, "--project-dir", str(setup_project_dir)],
        )

        # Assert
        # Verify CLI execution failed with correct exit code
        assert result.exit_code == 1, f"CLI should exit with code 1 for missing dataset, got: {result.output}"

        # Verify ProjectWrapper.delete_dataset was called
        mock_delete_dataset.assert_called_once_with(nonexistent_dataset, False)

        # Verify CLI output contains error messages
        assert "Failed to delete" in result.output, "Should display failure message"
        assert nonexistent_dataset in result.output, "Should mention the dataset name that failed"
        assert "does not exist" in result.output, "Should display specific error message about missing dataset"

    @pytest.mark.integration
    def test_delete_dataset_handles_no_such_dataset_error(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete dataset command properly handles NoSuchDatasetError.

        This test verifies that the CLI properly handles the standard NoSuchDatasetError
        exception that may be raised by ProjectWrapper when a dataset is not found.
        """
        # Arrange
        nonexistent_dataset = "missing_research_dataset"
        expected_error_message = f"Dataset '{nonexistent_dataset}' not found in project"

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_dataset = mocker.patch.object(
            ProjectWrapper,
            "delete_dataset",
            side_effect=ProjectWrapper.NoSuchDatasetError(expected_error_message),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "dataset", nonexistent_dataset, "--project-dir", str(setup_project_dir)],
        )

        # Assert
        # Verify CLI execution failed with correct exit code
        assert result.exit_code == 1, f"CLI should exit with code 1 for missing dataset, got: {result.output}"

        # Verify ProjectWrapper.delete_dataset was called
        mock_delete_dataset.assert_called_once_with(nonexistent_dataset, False)

        # Verify CLI output contains error messages
        assert "Failed to delete" in result.output, "Should display failure message"
        assert nonexistent_dataset in result.output, "Should mention the dataset name that failed"
        assert "not found in project" in result.output, "Should display specific error message from exception"


class TestDeleteCommandDryRun:
    """Test dry-run functionality across different delete commands."""

    @pytest.mark.integration
    def test_delete_pipeline_with_dry_run_flag_propagation(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test that --dry-run flag is properly propagated to pipeline deletion.

        This integration test verifies that when the --dry-run flag is specified
        on a pipeline delete command, it:
        - Correctly parses the command line flag
        - Passes dry_run=True to ProjectWrapper.delete_pipeline method
        - Still displays success messages as if the operation completed
        - Exits with success code 0
        """
        # Arrange
        pipeline_names = ["test_processing_pipeline"]
        expected_path = Path("/project/pipelines/test_processing_pipeline")

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_pipeline = mocker.patch.object(
            ProjectWrapper,
            "delete_pipeline",
            return_value=expected_path,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "pipeline", *pipeline_names, "--project-dir", str(setup_project_dir), "--dry-run"],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed with dry-run, got: {result.output}"

        # Verify dry_run=True was passed to delete_pipeline method
        mock_delete_pipeline.assert_called_once_with("test_processing_pipeline", True)

        # Verify CLI output contains success message
        assert "Deleted" in result.output, "Should display success message even in dry-run mode"
        assert "test_processing_pipeline" in result.output, "Should mention the pipeline name"
        assert str(expected_path) in result.output, "Should show the pipeline path"

    @pytest.mark.integration
    def test_delete_collection_with_dry_run_flag_propagation(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test that --dry-run flag is properly propagated to collection deletion.

        This test ensures consistency of dry-run behavior across different
        delete command types by verifying collection deletion.
        """
        # Arrange
        collection_names = ["marine_survey_data"]
        expected_path = Path("/project/collections/marine_survey_data")

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_collection = mocker.patch.object(
            ProjectWrapper,
            "delete_collection",
            return_value=expected_path,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "collection", *collection_names, "--project-dir", str(setup_project_dir), "--dry-run"],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed with dry-run, got: {result.output}"

        # Verify dry_run=True was passed to delete_collection method
        mock_delete_collection.assert_called_once_with("marine_survey_data", True)

        # Verify CLI output contains success message
        assert "Deleted" in result.output, "Should display success message even in dry-run mode"
        assert "marine_survey_data" in result.output, "Should mention the collection name"

    @pytest.mark.integration
    def test_delete_target_with_dry_run_flag_propagation(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test that --dry-run flag is properly propagated to target deletion.

        This test verifies dry-run functionality for distribution target deletion,
        ensuring consistent behavior across all delete command types.
        """
        # Arrange
        target_names = ["s3_distribution_target"]
        expected_path = Path("/project/targets/s3_distribution_target")

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_target = mocker.patch.object(
            ProjectWrapper,
            "delete_target",
            return_value=expected_path,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "target", *target_names, "--project-dir", str(setup_project_dir), "--dry-run"],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed with dry-run, got: {result.output}"

        # Verify dry_run=True was passed to delete_target method
        mock_delete_target.assert_called_once_with("s3_distribution_target", True)

        # Verify CLI output contains success message
        assert "Deleted" in result.output, "Should display success message even in dry-run mode"
        assert "s3_distribution_target" in result.output, "Should mention the target name"


class TestDeletePipelineCommand:
    """Test CLI delete pipeline command integration."""

    @pytest.mark.integration
    def test_delete_multiple_pipelines_with_partial_failures(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete pipeline command with mixed success and failure results.

        This integration test verifies that when deleting multiple pipelines where some exist
        and others don't, the CLI properly:
        - Processes all pipelines in the batch operation
        - Shows success messages for existing pipelines
        - Shows error messages for missing pipelines
        - Exits with error code 1 due to failures
        - Maintains processing order and handles each item appropriately
        """
        # Arrange
        pipeline_names = ["data_processing_pipeline", "missing_pipeline", "analysis_pipeline"]
        existing_path_1 = Path("/project/pipelines/data_processing_pipeline")
        existing_path_2 = Path("/project/pipelines/analysis_pipeline")
        missing_error = "Pipeline 'missing_pipeline' not found in project"

        def mock_delete_side_effect(name: str, _dry_run: bool) -> Path:
            if name == "missing_pipeline":
                raise ProjectWrapper.NoSuchPipelineError(missing_error)
            if name == "data_processing_pipeline":
                return existing_path_1
            if name == "analysis_pipeline":
                return existing_path_2
            unexpected_name_msg = f"Unexpected pipeline name: {name}"
            raise ValueError(unexpected_name_msg)

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_pipeline = mocker.patch.object(
            ProjectWrapper,
            "delete_pipeline",
            side_effect=mock_delete_side_effect,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "pipeline", *pipeline_names, "--project-dir", str(setup_project_dir)],
        )

        # Assert
        # Verify CLI execution failed due to partial failures
        assert result.exit_code == 1, f"CLI should exit with code 1 for partial failures, got: {result.output}"

        # Verify all pipelines were attempted
        assert mock_delete_pipeline.call_count == 3, "Should attempt to delete all 3 pipelines"
        mock_delete_pipeline.assert_any_call("data_processing_pipeline", False)
        mock_delete_pipeline.assert_any_call("missing_pipeline", False)
        mock_delete_pipeline.assert_any_call("analysis_pipeline", False)

        # Verify CLI output contains both success and error messages
        assert "Deleted" in result.output, "Should show success messages for existing pipelines"
        assert "Failed to delete" in result.output, "Should show failure message for missing pipeline"
        assert "data_processing_pipeline" in result.output, "Should mention successful pipeline"
        assert "analysis_pipeline" in result.output, "Should mention other successful pipeline"
        assert "missing_pipeline" in result.output, "Should mention failed pipeline"
        assert "not found" in result.output, "Should display specific error message"

    @pytest.mark.integration
    def test_delete_multiple_pipelines_successful(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test successful deletion of multiple pipelines via CLI command.

        This integration test verifies that the delete pipeline CLI command correctly:
        - Processes multiple pipeline names from command line arguments
        - Calls the ProjectWrapper.delete_pipeline method for each pipeline
        - Displays success messages for each deleted pipeline
        - Exits with code 0 on successful completion
        """
        # Arrange
        pipeline_names = ["image_processing_pipeline", "data_analysis_pipeline"]
        expected_paths = [
            Path("/project/pipelines/image_processing_pipeline"),
            Path("/project/pipelines/data_analysis_pipeline"),
        ]

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_pipeline = mocker.patch.object(
            ProjectWrapper,
            "delete_pipeline",
            side_effect=expected_paths,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "pipeline", *pipeline_names, "--project-dir", str(setup_project_dir)],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed, got: {result.output}"

        # Verify ProjectWrapper.delete_pipeline was called correctly for each pipeline
        assert mock_delete_pipeline.call_count == 2, "Should call delete_pipeline twice"
        mock_delete_pipeline.assert_any_call("image_processing_pipeline", False)
        mock_delete_pipeline.assert_any_call("data_analysis_pipeline", False)

        # Verify CLI output contains success messages for each pipeline
        assert "Deleted" in result.output, "Should display success message"
        assert "image_processing_pipeline" in result.output, "Should mention first pipeline name"
        assert "data_analysis_pipeline" in result.output, "Should mention second pipeline name"
        assert str(expected_paths[0]) in result.output, "Should show first pipeline path"
        assert str(expected_paths[1]) in result.output, "Should show second pipeline path"

    @pytest.mark.integration
    def test_delete_pipeline_with_dry_run_flag(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete pipeline command with dry-run flag passes correct parameter.

        This test ensures that the --dry-run flag is properly propagated through
        the CLI to the underlying delete_pipeline method.
        """
        # Arrange
        pipeline_names = ["test_pipeline"]
        expected_path = Path("/project/pipelines/test_pipeline")

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_pipeline = mocker.patch.object(
            ProjectWrapper,
            "delete_pipeline",
            return_value=expected_path,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "pipeline", *pipeline_names, "--project-dir", str(setup_project_dir), "--dry-run"],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed with dry-run, got: {result.output}"

        # Verify delete_pipeline was called with dry_run=True
        mock_delete_pipeline.assert_called_once_with("test_pipeline", True)

        # Verify CLI output contains success message
        assert "Deleted" in result.output, "Should display success message even in dry-run"
        assert "test_pipeline" in result.output, "Should mention pipeline name"
