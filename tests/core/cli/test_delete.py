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
    def test_batch_delete_operation_handles_partial_failures(self) -> None:
        """Test that batch_delete_operation correctly handles mixed success and failure outcomes.

        This test verifies that when some items succeed and others fail during batch deletion,
        the function properly separates successful operations from failed ones and preserves
        the exact error messages from exceptions. It also verifies that processing continues
        for remaining items even when failures occur.
        """
        # Arrange
        items = ["successful_item1", "failing_item", "successful_item2"]
        expected_error_message = "Collection 'failing_item' not found in project"

        def mock_delete_func(name: str, dry_run: bool) -> Path:
            # Verify dry_run parameter is passed correctly
            assert isinstance(dry_run, bool), "dry_run should be a boolean"
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
        # Verify return types
        assert isinstance(success_items, list), "Should return success_items as list"
        assert isinstance(errors, list), "Should return errors as list"

        # Verify counts
        assert len(success_items) == 2, "Should have exactly 2 successful operations"
        assert len(errors) == 1, "Should have exactly 1 failed operation"

        # Verify total items processed
        total_processed = len(success_items) + len(errors)
        assert total_processed == len(items), "Should process all input items"

        # Verify error details
        assert errors[0] == (
            "failing_item",
            expected_error_message,
        ), "Error should contain exact item name and error message"

        # Verify error tuple structure
        error_name, error_msg = errors[0]
        assert isinstance(error_name, str), "Error name should be string"
        assert isinstance(error_msg, str), "Error message should be string"

        # Verify successful operations (order preserved)
        expected_success_items = [
            ("successful_item1", Path("/mock/path/to/successful_item1")),
            ("successful_item2", Path("/mock/path/to/successful_item2")),
        ]
        assert (
            success_items == expected_success_items
        ), "Success items should contain exact names and paths in processing order"

        # Verify success item tuple structure
        for item_name, item_path in success_items:
            assert isinstance(item_name, str), "Success item name should be string"
            assert isinstance(item_path, Path), "Success item path should be Path object"

    @pytest.mark.unit
    def test_batch_delete_operation_preserves_processing_order(self) -> None:
        """Test that batch_delete_operation preserves the order of successful operations.

        This verifies that successful items are returned in the same order they were
        processed, even when there are failures interspersed. It specifically tests:
        - Successful items maintain their relative order from input
        - Failed items maintain their relative order from input
        - Processing order is preserved despite mixed success/failure outcomes
        """
        # Arrange
        items = ["first", "second_fails", "third", "fourth_fails", "fifth"]

        def mock_delete_func(name: str, dry_run: bool) -> Path:
            # Verify dry_run parameter is passed correctly
            assert isinstance(dry_run, bool), "dry_run should be a boolean"
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
        # Verify return types
        assert isinstance(success_items, list), "Should return success_items as list"
        assert isinstance(errors, list), "Should return errors as list"

        # Verify counts
        assert len(success_items) == 3, "Should have 3 successful operations"
        assert len(errors) == 2, "Should have 2 failed operations"

        # Verify total items processed
        total_processed = len(success_items) + len(errors)
        assert total_processed == len(items), "Should process all input items"

        # Verify successful operations maintain order
        expected_successful_names = ["first", "third", "fifth"]
        actual_successful_names = [name for name, _ in success_items]
        assert (
            actual_successful_names == expected_successful_names
        ), "Successful items should maintain their original processing order"

        # Verify successful items have correct paths and order
        expected_success_items = [
            ("first", Path("/order/test/first")),
            ("third", Path("/order/test/third")),
            ("fifth", Path("/order/test/fifth")),
        ]
        assert (
            success_items == expected_success_items
        ), "Success items should have correct paths and maintain processing order"

        # Verify failed operations maintain order
        expected_failed_names = ["second_fails", "fourth_fails"]
        actual_failed_names = [name for name, _ in errors]
        assert (
            actual_failed_names == expected_failed_names
        ), "Failed items should maintain their original processing order"

        # Verify failed items have correct error messages and order
        expected_error_messages = [
            "Failed to delete second_fails",
            "Failed to delete fourth_fails",
        ]
        actual_error_messages = [msg for _, msg in errors]
        assert (
            actual_error_messages == expected_error_messages
        ), "Error messages should match expected content and order"

        # Verify that the relative order from original input is preserved
        original_indices = {name: i for i, name in enumerate(items)}
        success_original_indices = [original_indices[name] for name, _ in success_items]
        error_original_indices = [original_indices[name] for name, _ in errors]

        # Success indices should be in ascending order (0, 2, 4)
        assert success_original_indices == sorted(
            success_original_indices,
        ), "Success items should maintain their relative order from original input"

        # Error indices should be in ascending order (1, 3)
        assert error_original_indices == sorted(
            error_original_indices,
        ), "Error items should maintain their relative order from original input"


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


class TestDeleteProjectCommand:
    """Test CLI delete project command integration."""

    @pytest.mark.integration
    def test_delete_project_command(
        self,
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
        # Use the established helper function for CLI success assertions
        assert_cli_success(result, context="Project deletion command")

        # Verify delete_project was called
        mock_delete.assert_called_once()

        # Verify CLI output contains success message and project path
        assert "Deleted" in result.output, "Should display success message"
        assert str(expected_deleted_path) in result.output, "Should show the deleted project path"

    @pytest.mark.integration
    def test_delete_project_invalid_structure(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete project command properly handles InvalidStructureError.

        This integration test verifies that when attempting to delete a project
        with an invalid directory structure, the CLI command:
        - Catches the InvalidStructureError from ProjectWrapper initialization
        - Displays appropriate error message indicating the project is not valid
        - Exits with error code 1
        - Shows the specific error context about invalid structure
        - Does not display any success messages for the failed operation

        Uses focused mocking to test the exception handling path without over-mocking business logic.
        """
        # Arrange
        expected_error_message = "Invalid project structure detected"

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)

        # Mock only the specific method that would detect invalid structure
        # This allows testing the error handling path without over-mocking
        mocker.patch.object(
            ProjectWrapper,
            "_check_file_structure",
            side_effect=ProjectWrapper.InvalidStructureError(expected_error_message),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "project", "--project-dir", str(setup_project_dir)],
        )

        # Assert
        # Use the established helper function for CLI failure assertions
        assert_cli_failure(
            result,
            expected_error="not valid project",
            expected_exit_code=1,
            context="Project deletion with invalid structure",
        )

        # Verify CLI output contains specific error messages
        assert "Marimba" in result.output, "Should mention Marimba in the error message"
        assert setup_project_dir.name in result.output, "Should mention the project directory name"

        # Verify that no success messages are shown for failed operation
        assert "Deleted" not in result.output, "Should not display success message for failed operation"

    @pytest.mark.integration
    def test_delete_project_dry_run(
        self,
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
        - Shows no error messages for successful dry-run operations
        - Displays properly formatted success message with project path
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
        # Use the established helper function for CLI success assertions
        assert_cli_success(result, context="Project deletion with dry-run flag")

        # Verify ProjectWrapper was initialized with dry_run=True (most important assertion for dry-run test)
        mock_project_wrapper_class.assert_called_once_with(setup_project_dir, dry_run=True)

        # Verify delete_project was called exactly once (dry-run logic is handled within the ProjectWrapper)
        mock_project_wrapper_instance.delete_project.assert_called_once()

        # Verify CLI output contains success message and project details
        assert "Deleted" in result.output, "Should display success message even in dry-run mode"
        assert str(expected_deleted_path) in result.output, "Should show the project path"

        # Verify no error messages appear for successful dry-run operation
        assert "Failed" not in result.output, "Should not display error messages for successful dry-run"
        assert "Error" not in result.output, "Should not display error messages for successful dry-run"

        # Verify specific success message format
        assert "Marimba project" in result.output, "Should display formatted success message with project identifier"


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
        assert mock_delete_collection.call_count == 2, "Should call delete_collection exactly twice"
        mock_delete_collection.assert_any_call("marine_data", False)
        mock_delete_collection.assert_any_call("coastal_survey", False)

        # Verify calls were made with correct dry_run parameter (False by default)
        for call in mock_delete_collection.call_args_list:
            assert call[0][1] is False, "Should call delete_collection with dry_run=False by default"

        # Verify CLI output contains success messages for each collection
        assert "Deleted" in result.output, "Should display success message"
        assert "marine_data" in result.output, "Should mention first collection name"
        assert "coastal_survey" in result.output, "Should mention second collection name"
        assert str(expected_paths[0]) in result.output, "Should show first collection path"
        assert str(expected_paths[1]) in result.output, "Should show second collection path"

        # Verify no error messages appear for successful operations
        assert "Failed to delete" not in result.output, "Should not display error messages for successful operations"

        # Verify success message format for each collection
        for collection_name in collection_names:
            assert "Deleted" in result.output, "Should display success message for each collection"
            assert (
                f'collection "{collection_name}"' in result.output
            ), f"Should display formatted success message with collection name {collection_name}"

    @pytest.mark.integration
    def test_delete_collection_with_dry_run_flag(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete collection command with dry-run flag passes correct parameter.

        This integration test verifies that the delete collection CLI command correctly:
        - Parses the --dry-run flag from command line arguments
        - Passes dry_run=True through batch_delete_operation to delete_collection method
        - Still displays success messages as if the operation completed
        - Exits with success code 0
        - Shows the specific collection path in output
        """
        # Arrange
        collection_names = ["test_collection"]
        expected_path = Path("/project/collections/test_collection")

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)

        # Mock batch_delete_operation to verify dry_run parameter propagation
        mock_batch_delete = mocker.patch(
            "marimba.core.cli.delete.batch_delete_operation",
            return_value=([(collection_names[0], expected_path)], []),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "collection", *collection_names, "--project-dir", str(setup_project_dir), "--dry-run"],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed with dry-run, got: {result.output}"

        # Verify batch_delete_operation was called with dry_run=True (most important for dry-run test)
        mock_batch_delete.assert_called_once()
        call_args = mock_batch_delete.call_args
        assert call_args[0][0] == collection_names, "Should pass collection names"
        assert call_args[0][2] == "collection", "Should pass entity type"
        assert call_args[0][3] == "Deleting collections...", "Should pass description"
        assert call_args[0][4] is True, "Should pass dry_run=True"

        # Verify CLI output contains success message and collection details
        assert "Deleted" in result.output, "Should display success message even in dry-run mode"
        assert "test_collection" in result.output, "Should mention collection name"
        assert str(expected_path) in result.output, "Should show the collection path"

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

        # Verify ProjectWrapper.delete_collection was called with correct parameters
        mock_delete_collection.assert_called_once_with(nonexistent_collection, False)

        # Verify that exactly one call was made (no retries or duplicates)
        assert mock_delete_collection.call_count == 1, "Should call delete_collection exactly once"

        # Verify CLI output contains error messages
        assert "Failed to delete" in result.output, "Should display failure message"
        assert nonexistent_collection in result.output, "Should mention the collection name that failed"
        assert "not found in project" in result.output, "Should display the specific error message from exception"

        # Verify that no success messages are shown for failed operation
        assert "Deleted" not in result.output, "Should not display success message for failed operation"

        # Verify error message format follows expected pattern
        assert (
            f'Failed to delete collection "{nonexistent_collection}"' in result.output
        ), "Should display formatted error message with collection name"

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

        def mock_delete_side_effect(name: str, dry_run: bool) -> Path:
            # Verify dry_run parameter is passed correctly
            assert isinstance(dry_run, bool), "dry_run should be a boolean"
            assert dry_run is False, "Should pass dry_run=False by default"

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

        # Verify all calls were made with correct dry_run parameter
        for call in mock_delete_collection.call_args_list:
            assert call[0][1] is False, "All calls should use dry_run=False by default"

        # Verify CLI output contains both success and error messages
        assert "Deleted" in result.output, "Should show success messages for existing collections"
        assert "Failed to delete" in result.output, "Should show failure message for missing collection"
        assert "existing_collection" in result.output, "Should mention successful collection"
        assert "another_existing" in result.output, "Should mention other successful collection"
        assert "missing_collection" in result.output, "Should mention failed collection"
        assert "not found" in result.output, "Should display specific error message"

        # Verify successful collection paths appear in output
        assert str(existing_path) in result.output, "Should show path for first successful collection"
        assert str(another_existing_path) in result.output, "Should show path for second successful collection"

        # Verify specific success message format for successful collections
        assert (
            'collection "existing_collection"' in result.output
        ), "Should show formatted success message for first collection"
        assert (
            'collection "another_existing"' in result.output
        ), "Should show formatted success message for second collection"

        # Verify specific error message format for failed collection
        assert (
            'Failed to delete collection "missing_collection"' in result.output
        ), "Should show formatted error message"

        # Verify that batch processing continued after failure (resilience test)
        successful_collections = ["existing_collection", "another_existing"]
        for collection in successful_collections:
            assert "Deleted" in result.output, f"Should show success for {collection} despite other failures"


class TestDeleteTargetCommand:
    """Test CLI delete target command integration."""

    @pytest.mark.integration
    def test_delete_multiple_targets_successful(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test successful deletion of multiple targets via CLI command.

        This integration test verifies that the delete target CLI command correctly:
        - Processes multiple target names from command line arguments
        - Calls the ProjectWrapper.delete_target method for each target
        - Displays success messages for each deleted target
        - Exits with code 0 on successful completion
        """
        # Arrange
        target_names = ["s3_target", "dap_server_target"]
        expected_paths = [
            Path("/project/targets/s3_target"),
            Path("/project/targets/dap_server_target"),
        ]

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_target = mocker.patch.object(
            ProjectWrapper,
            "delete_target",
            side_effect=expected_paths,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "target", *target_names, "--project-dir", str(setup_project_dir)],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed, got: {result.output}"

        # Verify ProjectWrapper.delete_target was called correctly for each target
        assert mock_delete_target.call_count == 2, "Should call delete_target exactly twice"
        mock_delete_target.assert_any_call("s3_target", False)
        mock_delete_target.assert_any_call("dap_server_target", False)

        # Verify calls were made with correct dry_run parameter (False by default)
        for call in mock_delete_target.call_args_list:
            assert call[0][1] is False, "Should call delete_target with dry_run=False by default"

        # Verify CLI output contains success messages for each target
        assert "Deleted" in result.output, "Should display success message"
        assert "s3_target" in result.output, "Should mention first target name"
        assert "dap_server_target" in result.output, "Should mention second target name"
        assert str(expected_paths[0]) in result.output, "Should show first target path"
        assert str(expected_paths[1]) in result.output, "Should show second target path"

        # Verify no error messages appear for successful operations
        assert "Failed to delete" not in result.output, "Should not display error messages for successful operations"

        # Verify success message format for each target
        for target_name in target_names:
            assert "Deleted" in result.output, "Should display success message for each target"
            assert (
                f'target "{target_name}"' in result.output
            ), f"Should display formatted success message with target name {target_name}"

    @pytest.mark.integration
    def test_delete_target_handles_nonexistent_target_error(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete target command properly handles NoSuchTargetError.

        This integration test verifies that when attempting to delete a target
        that doesn't exist, the CLI command:
        - Catches the NoSuchTargetError from ProjectWrapper
        - Displays appropriate error message with target name
        - Exits with error code 1
        - Shows the specific error message from the exception
        """
        # Arrange
        nonexistent_target = "missing_s3_target"
        expected_error_message = f"Target '{nonexistent_target}' not found in project"

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_target = mocker.patch.object(
            ProjectWrapper,
            "delete_target",
            side_effect=ProjectWrapper.NoSuchTargetError(expected_error_message),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "target", nonexistent_target, "--project-dir", str(setup_project_dir)],
        )

        # Assert
        # Verify CLI execution failed with correct exit code
        assert result.exit_code == 1, f"CLI should exit with code 1 for missing target, got: {result.output}"

        # Verify ProjectWrapper.delete_target was called with correct parameters
        mock_delete_target.assert_called_once_with(nonexistent_target, False)

        # Verify that exactly one call was made (no retries or duplicates)
        assert mock_delete_target.call_count == 1, "Should call delete_target exactly once"

        # Verify CLI output contains error messages
        assert "Failed to delete" in result.output, "Should display failure message"
        assert nonexistent_target in result.output, "Should mention the target name that failed"
        assert "not found" in result.output, "Should display the specific error message from exception"

        # Verify that no success messages are shown for failed operation
        assert "Deleted" not in result.output, "Should not display success message for failed operation"

        # Verify error message format follows expected pattern
        assert (
            f'Failed to delete target "{nonexistent_target}"' in result.output
        ), "Should display formatted error message with target name"

    @pytest.mark.integration
    def test_delete_target_with_dry_run_flag(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete target command with dry-run flag passes correct parameter.

        This integration test verifies that the delete target CLI command correctly:
        - Parses the --dry-run flag from command line arguments
        - Passes dry_run=True through batch_delete_operation to delete_target method
        - Still displays success messages as if the operation completed
        - Exits with success code 0
        - Shows the specific target path in output
        """
        # Arrange
        target_names = ["test_target"]
        expected_path = Path("/project/targets/test_target")

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)

        # Mock batch_delete_operation to verify dry_run parameter propagation
        mock_batch_delete = mocker.patch(
            "marimba.core.cli.delete.batch_delete_operation",
            return_value=([(target_names[0], expected_path)], []),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "target", *target_names, "--project-dir", str(setup_project_dir), "--dry-run"],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed with dry-run, got: {result.output}"

        # Verify batch_delete_operation was called with dry_run=True (most important for dry-run test)
        mock_batch_delete.assert_called_once()
        call_args = mock_batch_delete.call_args
        assert call_args[0][0] == target_names, "Should pass target names"
        assert call_args[0][2] == "target", "Should pass entity type"
        assert call_args[0][3] == "Deleting targets...", "Should pass description"
        assert call_args[0][4] is True, "Should pass dry_run=True"

        # Verify CLI output contains success message and target details
        assert "Deleted" in result.output, "Should display success message even in dry-run mode"
        assert "test_target" in result.output, "Should mention target name"
        assert str(expected_path) in result.output, "Should show the target path"

    @pytest.mark.integration
    def test_delete_multiple_targets_with_partial_failures(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete target command with mixed success and failure results.

        This test verifies that when deleting multiple targets where some exist
        and others don't, the CLI properly:
        - Processes all targets in the batch
        - Shows success messages for existing targets
        - Shows error messages for missing targets
        - Exits with error code 1 due to failures
        """
        # Arrange
        target_names = ["existing_target", "missing_target", "another_existing"]
        existing_path = Path("/project/targets/existing_target")
        another_existing_path = Path("/project/targets/another_existing")
        missing_error = "Target 'missing_target' not found"

        def mock_delete_side_effect(name: str, dry_run: bool) -> Path:
            # Verify dry_run parameter is passed correctly
            assert isinstance(dry_run, bool), "dry_run should be a boolean"
            assert dry_run is False, "Should pass dry_run=False by default"

            if name == "missing_target":
                raise ProjectWrapper.NoSuchTargetError(missing_error)
            if name == "existing_target":
                return existing_path
            if name == "another_existing":
                return another_existing_path
            unexpected_name_msg = f"Unexpected target name: {name}"
            raise ValueError(unexpected_name_msg)

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)
        mock_delete_target = mocker.patch.object(
            ProjectWrapper,
            "delete_target",
            side_effect=mock_delete_side_effect,
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "target", *target_names, "--project-dir", str(setup_project_dir)],
        )

        # Assert
        # Verify CLI execution failed due to partial failures
        assert result.exit_code == 1, f"CLI should exit with code 1 for partial failures, got: {result.output}"

        # Verify all targets were attempted
        assert mock_delete_target.call_count == 3, "Should attempt to delete all 3 targets"
        mock_delete_target.assert_any_call("existing_target", False)
        mock_delete_target.assert_any_call("missing_target", False)
        mock_delete_target.assert_any_call("another_existing", False)

        # Verify all calls were made with correct dry_run parameter
        for call in mock_delete_target.call_args_list:
            assert call[0][1] is False, "All calls should use dry_run=False by default"

        # Verify CLI output contains both success and error messages
        assert "Deleted" in result.output, "Should show success messages for existing targets"
        assert "Failed to delete" in result.output, "Should show failure message for missing target"
        assert "existing_target" in result.output, "Should mention successful target"
        assert "another_existing" in result.output, "Should mention other successful target"
        assert "missing_target" in result.output, "Should mention failed target"
        assert "not found" in result.output, "Should display specific error message"

        # Verify successful target paths appear in output
        assert str(existing_path) in result.output, "Should show path for first successful target"
        assert str(another_existing_path) in result.output, "Should show path for second successful target"

        # Verify specific success message format for successful targets
        assert 'target "existing_target"' in result.output, "Should show formatted success message for first target"
        assert 'target "another_existing"' in result.output, "Should show formatted success message for second target"

        # Verify specific error message format for failed target
        assert 'Failed to delete target "missing_target"' in result.output, "Should show formatted error message"

        # Verify that batch processing continued after failure (resilience test)
        successful_targets = ["existing_target", "another_existing"]
        for target in successful_targets:
            assert "Deleted" in result.output, f"Should show success for {target} despite other failures"


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
        assert mock_delete_dataset.call_count == 2, "Should call delete_dataset exactly twice"
        mock_delete_dataset.assert_any_call("marine_analysis_2023", False)
        mock_delete_dataset.assert_any_call("coastal_survey_results", False)

        # Verify calls were made with correct dry_run parameter (False by default)
        for call in mock_delete_dataset.call_args_list:
            assert call[0][1] is False, "Should call delete_dataset with dry_run=False by default"

        # Verify CLI output contains success messages for each dataset
        assert "Deleted" in result.output, "Should display success message"
        assert "marine_analysis_2023" in result.output, "Should mention first dataset name"
        assert "coastal_survey_results" in result.output, "Should mention second dataset name"
        assert str(expected_paths[0]) in result.output, "Should show first dataset path"
        assert str(expected_paths[1]) in result.output, "Should show second dataset path"

        # Verify no error messages appear for successful operations
        assert "Failed to delete" not in result.output, "Should not display error messages for successful operations"

        # Verify success message format for each dataset
        for dataset_name in dataset_names:
            assert "Deleted" in result.output, "Should display success message for each dataset"
            assert (
                f'dataset "{dataset_name}"' in result.output
            ), f"Should display formatted success message with dataset name {dataset_name}"

    @pytest.mark.integration
    def test_delete_dataset_with_dry_run_flag(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete dataset command with dry-run flag passes correct parameter.

        This integration test verifies that the delete dataset CLI command correctly:
        - Parses the --dry-run flag from command line arguments
        - Passes dry_run=True through batch_delete_operation to delete_dataset method
        - Still displays success messages as if the operation completed
        - Exits with success code 0
        - Shows the specific dataset path in output
        """
        # Arrange
        dataset_names = ["test_dataset"]
        expected_path = Path("/project/datasets/test_dataset")

        mocker.patch("marimba.core.cli.delete.find_project_dir_or_exit", return_value=setup_project_dir)

        # Mock batch_delete_operation to verify dry_run parameter propagation
        mock_batch_delete = mocker.patch(
            "marimba.core.cli.delete.batch_delete_operation",
            return_value=([(dataset_names[0], expected_path)], []),
        )

        # Act
        result = runner.invoke(
            marimba_cli,
            ["delete", "dataset", *dataset_names, "--project-dir", str(setup_project_dir), "--dry-run"],
        )

        # Assert
        # Verify CLI execution succeeded
        assert result.exit_code == 0, f"CLI command should succeed with dry-run, got: {result.output}"

        # Verify batch_delete_operation was called with dry_run=True (most important for dry-run test)
        mock_batch_delete.assert_called_once()
        call_args = mock_batch_delete.call_args
        assert call_args[0][0] == dataset_names, "Should pass dataset names"
        assert call_args[0][2] == "dataset", "Should pass entity type"
        assert call_args[0][3] == "Deleting datasets...", "Should pass description"
        assert call_args[0][4] is True, "Should pass dry_run=True"

        # Verify CLI output contains success message and dataset details
        assert "Deleted" in result.output, "Should display success message even in dry-run mode"
        assert "test_dataset" in result.output, "Should mention dataset name"
        assert str(expected_path) in result.output, "Should show the dataset path"

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

        # Verify ProjectWrapper.delete_dataset was called with correct parameters
        mock_delete_dataset.assert_called_once_with(nonexistent_dataset, False)

        # Verify that exactly one call was made (no retries or duplicates)
        assert mock_delete_dataset.call_count == 1, "Should call delete_dataset exactly once"

        # Verify CLI output contains error messages
        assert "Failed to delete" in result.output, "Should display failure message"
        assert nonexistent_dataset in result.output, "Should mention the dataset name that failed"
        assert "does not exist" in result.output, "Should display specific error message about missing dataset"

        # Verify that no success messages are shown for failed operation
        assert "Deleted" not in result.output, "Should not display success message for failed operation"

        # Verify error message format follows expected pattern
        assert (
            f'Failed to delete dataset "{nonexistent_dataset}"' in result.output
        ), "Should display formatted error message with dataset name"

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

        # Verify ProjectWrapper.delete_dataset was called with correct parameters
        mock_delete_dataset.assert_called_once_with(nonexistent_dataset, False)

        # Verify that exactly one call was made (no retries or duplicates)
        assert mock_delete_dataset.call_count == 1, "Should call delete_dataset exactly once"

        # Verify CLI output contains error messages
        assert "Failed to delete" in result.output, "Should display failure message"
        assert nonexistent_dataset in result.output, "Should mention the dataset name that failed"
        assert "not found in project" in result.output, "Should display specific error message from exception"

        # Verify that no success messages are shown for failed operation
        assert "Deleted" not in result.output, "Should not display success message for failed operation"

        # Verify error message format follows expected pattern
        assert (
            f'Failed to delete dataset "{nonexistent_dataset}"' in result.output
        ), "Should display formatted error message with dataset name"


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
        - Correctly parses the command line flag from CLI arguments
        - Passes dry_run=True to ProjectWrapper.delete_pipeline method
        - Still displays success messages as if the operation completed
        - Exits with success code 0
        - Shows no error messages for successful dry-run operations
        - Uses batch_delete_operation to coordinate the dry-run process
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

        # Verify dry_run=True was passed to delete_pipeline method (most critical assertion for dry-run test)
        mock_delete_pipeline.assert_called_once_with("test_processing_pipeline", True)

        # Verify that exactly one call was made (no retries or duplicates)
        assert mock_delete_pipeline.call_count == 1, "Should call delete_pipeline exactly once"

        # Verify CLI output contains success message and all expected details
        assert "Deleted" in result.output, "Should display success message even in dry-run mode"
        assert "test_processing_pipeline" in result.output, "Should mention the pipeline name"
        assert str(expected_path) in result.output, "Should show the pipeline path"

        # Verify no error messages appear for successful dry-run operation
        assert "Failed to delete" not in result.output, "Should not display error messages for successful dry-run"

        # Verify specific success message format
        assert (
            'pipeline "test_processing_pipeline"' in result.output
        ), "Should display formatted success message with pipeline name"

        # Verify that all expected output elements are present for complete dry-run verification
        expected_elements = ["Deleted", "pipeline", "test_processing_pipeline", str(expected_path)]
        for element in expected_elements:
            assert element in result.output, f"Dry-run output should contain '{element}' for complete verification"

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
        - Continues processing after failures (resilience testing)
        """
        # Arrange
        pipeline_names = ["data_processing_pipeline", "missing_pipeline", "analysis_pipeline"]
        existing_path_1 = Path("/project/pipelines/data_processing_pipeline")
        existing_path_2 = Path("/project/pipelines/analysis_pipeline")
        missing_error = "Pipeline 'missing_pipeline' not found in project"

        def mock_delete_side_effect(name: str, dry_run: bool) -> Path:
            # Verify dry_run parameter is passed correctly
            assert isinstance(dry_run, bool), "dry_run should be a boolean"
            assert dry_run is False, "Should pass dry_run=False by default"

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

        # Verify all calls were made with correct dry_run parameter
        for call in mock_delete_pipeline.call_args_list:
            assert call[0][1] is False, "All calls should use dry_run=False by default"

        # Verify CLI output contains both success and error messages
        assert "Deleted" in result.output, "Should show success messages for existing pipelines"
        assert "Failed to delete" in result.output, "Should show failure message for missing pipeline"
        assert "data_processing_pipeline" in result.output, "Should mention successful pipeline"
        assert "analysis_pipeline" in result.output, "Should mention other successful pipeline"
        assert "missing_pipeline" in result.output, "Should mention failed pipeline"
        assert "not found" in result.output, "Should display specific error message"

        # Verify successful pipeline paths appear in output
        assert str(existing_path_1) in result.output, "Should show path for first successful pipeline"
        assert str(existing_path_2) in result.output, "Should show path for second successful pipeline"

        # Verify specific success message format for successful pipelines
        assert (
            'pipeline "data_processing_pipeline"' in result.output
        ), "Should show formatted success message for first pipeline"
        assert (
            'pipeline "analysis_pipeline"' in result.output
        ), "Should show formatted success message for second pipeline"

        # Verify specific error message format for failed pipeline
        assert 'Failed to delete pipeline "missing_pipeline"' in result.output, "Should show formatted error message"

        # Verify that batch processing continued after failure (resilience test)
        successful_pipelines = ["data_processing_pipeline", "analysis_pipeline"]
        for pipeline in successful_pipelines:
            assert "Deleted" in result.output, f"Should show success for {pipeline} despite other failures"

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
        - Uses batch_delete_operation to coordinate the deletion process
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
        assert mock_delete_pipeline.call_count == 2, "Should call delete_pipeline exactly twice"
        mock_delete_pipeline.assert_any_call("image_processing_pipeline", False)
        mock_delete_pipeline.assert_any_call("data_analysis_pipeline", False)

        # Verify calls were made with correct dry_run parameter (False by default)
        for call in mock_delete_pipeline.call_args_list:
            assert call[0][1] is False, "Should call delete_pipeline with dry_run=False by default"

        # Verify CLI output contains success messages for each pipeline
        assert "Deleted" in result.output, "Should display success message"
        assert "image_processing_pipeline" in result.output, "Should mention first pipeline name"
        assert "data_analysis_pipeline" in result.output, "Should mention second pipeline name"
        assert str(expected_paths[0]) in result.output, "Should show first pipeline path"
        assert str(expected_paths[1]) in result.output, "Should show second pipeline path"

        # Verify no error messages appear for successful operations
        assert "Failed to delete" not in result.output, "Should not display error messages for successful operations"

        # Verify success message format for each pipeline
        for pipeline_name in pipeline_names:
            assert "Deleted" in result.output, "Should display success message for each pipeline"
            assert (
                f'pipeline "{pipeline_name}"' in result.output
            ), f"Should display formatted success message with pipeline name {pipeline_name}"

    @pytest.mark.integration
    def test_delete_pipeline_with_dry_run_flag(
        self,
        mocker: pytest_mock.MockerFixture,
        setup_project_dir: Path,
    ) -> None:
        """Test delete pipeline command with dry-run flag passes correct parameter and displays proper output.

        This integration test ensures that the --dry-run flag is properly propagated through
        the CLI to the underlying delete_pipeline method and verifies comprehensive output formatting.
        It specifically tests:
        - Dry-run flag propagation to ProjectWrapper.delete_pipeline
        - Success message formatting in dry-run mode
        - Path display in output
        - No error messages for successful dry-run operations
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

        # Verify delete_pipeline was called with dry_run=True (most critical assertion for dry-run test)
        mock_delete_pipeline.assert_called_once_with("test_pipeline", True)

        # Verify that exactly one call was made
        assert mock_delete_pipeline.call_count == 1, "Should call delete_pipeline exactly once"

        # Verify CLI output contains success message and details
        assert "Deleted" in result.output, "Should display success message even in dry-run mode"
        assert "test_pipeline" in result.output, "Should mention pipeline name"
        assert str(expected_path) in result.output, "Should show the pipeline path in output"

        # Verify no error messages appear for successful dry-run operation
        assert "Failed to delete" not in result.output, "Should not display error messages for successful dry-run"

        # Verify specific success message format
        assert (
            'pipeline "test_pipeline"' in result.output
        ), "Should display formatted success message with pipeline name"

        # Verify the complete success message format includes all expected elements
        expected_elements = ["Deleted", "pipeline", "test_pipeline", str(expected_path)]
        for element in expected_elements:
            assert element in result.output, f"Output should contain '{element}' for complete success message"

    @pytest.mark.integration
    def test_delete_pipeline_handles_nonexistent_pipeline_error(
        self,
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
        - Uses batch_delete_operation to coordinate the error handling
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

        # Verify ProjectWrapper.delete_pipeline was called with correct parameters
        mock_delete_pipeline.assert_called_once_with(nonexistent_pipeline, False)

        # Verify that exactly one call was made (no retries or duplicates)
        assert mock_delete_pipeline.call_count == 1, "Should call delete_pipeline exactly once"

        # Verify CLI output contains error messages
        assert "Failed to delete" in result.output, "Should display failure message"
        assert nonexistent_pipeline in result.output, "Should mention the pipeline name that failed"
        assert "not found in project" in result.output, "Should display the specific error message from exception"

        # Verify that no success messages are shown for failed operation
        assert "Deleted" not in result.output, "Should not display success message for failed operation"

        # Verify error message format follows expected pattern
        assert (
            f'Failed to delete pipeline "{nonexistent_pipeline}"' in result.output
        ), "Should display formatted error message with pipeline name"
