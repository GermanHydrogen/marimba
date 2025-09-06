import os
from pathlib import Path

import pytest
import pytest_mock
import typer
from typer.testing import CliRunner

from marimba.core.utils.paths import find_project_dir, find_project_dir_or_exit
from marimba.core.wrappers.project import ProjectWrapper
from marimba.main import marimba_cli
from tests.conftest import TestDataFactory, assert_project_structure_exists

runner = CliRunner()

# ---------------------------------------------------------------------------------------------------------------------#
# Testing find_project_dir()
# ---------------------------------------------------------------------------------------------------------------------#


@pytest.fixture
def setup_test_directory_structure(tmp_path: Path) -> Path:
    """
    Fixture to set up a test directory structure.

    Args:
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        The path to the root of the test directory structure.
    """
    root = tmp_path / "root"
    root.mkdir()
    (root / ".marimba").mkdir()

    subdir = root / "subdir"
    subdir.mkdir()

    return subdir


@pytest.mark.integration
def test_find_project_dir_from_project_root(
    setup_test_directory_structure: Path,
) -> None:
    """
    Test find_project_dir when starting from the project root directory.

    Args:
        setup_test_directory_structure: Path to the test directory structure.

    Returns:
        None
    """
    root_dir = setup_test_directory_structure.parent
    assert find_project_dir(root_dir) == root_dir


@pytest.mark.integration
def test_find_project_dir_from_subdir(setup_test_directory_structure: Path) -> None:
    """
    Test find_project_dir when starting from a subdirectory within the project.

    Args:
        setup_test_directory_structure: Path to the test directory structure.

    Returns:
        None
    """
    root_dir = setup_test_directory_structure.parent
    assert find_project_dir(setup_test_directory_structure) == root_dir


@pytest.mark.integration
def test_find_project_dir_no_project_root(tmp_path: Path) -> None:
    """
    Test find_project_dir when there is no project root directory.

    Args:
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        None
    """
    non_project_dir = tmp_path / "non_project"
    non_project_dir.mkdir()

    assert find_project_dir(non_project_dir) is None


@pytest.mark.integration
def test_find_project_dir_from_nested_subdir(
    setup_test_directory_structure: Path,
) -> None:
    """
    Test find_project_dir when starting from a deeply nested subdirectory within the project.

    Args:
        setup_test_directory_structure: Path to the test directory structure.

    Returns:
        None
    """
    nested_dir = setup_test_directory_structure / "nested" / "deeper"
    nested_dir.mkdir(parents=True)

    root_dir = setup_test_directory_structure.parent
    assert find_project_dir(nested_dir) == root_dir


@pytest.mark.integration
def test_find_project_dir_invalid_path() -> None:
    """
    Test find_project_dir when given an invalid path.

    Returns:
        None
    """
    invalid_path = "invalid_path"
    assert find_project_dir(invalid_path) is None


@pytest.mark.integration
def test_find_project_dir_non_existent_path() -> None:
    """
    Test find_project_dir when given a non-existent path.

    Returns:
        None
    """
    non_existent_path = Path("/non_existent_path")
    assert find_project_dir(non_existent_path) is None


@pytest.mark.integration
def test_find_project_dir_no_read_access(tmp_path: Path) -> None:
    """
    Test find_project_dir when the starting path has no read access.

    Args:
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        None
    """
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    (root_dir / ".marimba").mkdir()

    os.chmod(root_dir, 0o000)  # Remove read access
    try:
        assert find_project_dir(root_dir) is None
    finally:
        os.chmod(root_dir, 0o755)  # Restore permissions for cleanup


@pytest.mark.integration
def test_find_project_dir_marimba_is_file(tmp_path: Path) -> None:
    """
    Test find_project_dir when .marimba is a file, not a directory.

    Args:
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        None
    """
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    (root_dir / ".marimba").touch()  # Create .marimba as a file

    assert find_project_dir(root_dir) is None


# ---------------------------------------------------------------------------------------------------------------------#
# Testing find_project_dir_or_exit()
# ---------------------------------------------------------------------------------------------------------------------#


@pytest.mark.integration
def test_find_project_dir_or_exit_with_valid_project_dir(
    setup_test_directory_structure: Path,
) -> None:
    """
    Test find_project_dir_or_exit when given a valid project directory.

    Args:
        setup_test_directory_structure: Path to the test directory structure.

    Returns:
        None
    """
    root_dir = setup_test_directory_structure.parent
    result = find_project_dir_or_exit(root_dir)
    assert result == root_dir


@pytest.mark.integration
def test_find_project_dir_or_exit_with_none_project_dir(
    mocker: pytest_mock.MockerFixture,
    setup_test_directory_structure: Path,
) -> None:
    """
    Test find_project_dir_or_exit when no project directory is given (None).

    Args:
        setup_test_directory_structure: Path to the test directory structure.

    Returns:
        None
    """
    root_dir = setup_test_directory_structure.parent
    mocker.patch("marimba.core.cli.new.Path.cwd", return_value=root_dir)
    result = find_project_dir_or_exit()
    assert result == root_dir


@pytest.mark.integration
def test_find_project_dir_or_exit_with_invalid_project_dir(tmp_path: Path) -> None:
    """
    Test find_project_dir_or_exit when given an invalid project directory.

    Args:
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        None
    """
    non_project_dir = tmp_path / "non_project"
    non_project_dir.mkdir()

    with pytest.raises(typer.Exit):
        find_project_dir_or_exit(non_project_dir)


@pytest.mark.integration
def test_find_project_dir_or_exit_raises_exit(
    mocker: pytest_mock.MockerFixture,
    setup_test_directory_structure: Path,
) -> None:
    """
    Test find_project_dir_or_exit when no project root directory is found, raising typer.Exit.

    Args:
        setup_test_directory_structure: Path to the test directory structure.

    Returns:
        None
    """
    non_project_dir = setup_test_directory_structure / "non_project"
    non_project_dir.mkdir()

    mocker.patch("marimba.core.utils.paths.find_project_dir", return_value=None)
    with pytest.raises(typer.Exit):
        find_project_dir_or_exit(non_project_dir)


# def test_find_project_dir_or_exit_with_relative_path(setup_test_directory_structure: Path) -> None:
#     """
#     Test find_project_dir_or_exit with a relative path.
#
#     Args:
#         setup_test_directory_structure: Path to the test directory structure.
#
#     Returns:
#         None
#     """
#     relative_path = setup_test_directory_structure.relative_to(Path.cwd())
#     root_dir = setup_test_directory_structure.parent
#
#     with patch("marimba.core.cli.new.find_project_dir", return_value=root_dir):
#         result = find_project_dir_or_exit(relative_path)
#         assert result == root_dir


@pytest.mark.integration
def test_find_project_dir_or_exit_with_symlink(mocker: pytest_mock.MockerFixture, tmp_path: Path) -> None:
    """
    Test find_project_dir_or_exit when there are symbolic links in the path.

    Args:
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        None
    """
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    (root_dir / ".marimba").mkdir()

    subdir = root_dir / "subdir"
    subdir.mkdir()

    symlink_dir = tmp_path / "symlink"
    symlink_dir.symlink_to(subdir)

    mocker.patch("marimba.core.utils.paths.find_project_dir", return_value=root_dir)
    result = find_project_dir_or_exit(symlink_dir)
    assert result == root_dir


@pytest.mark.integration
def test_find_project_dir_or_exit_with_no_read_access(tmp_path: Path) -> None:
    """
    Test find_project_dir_or_exit when the starting path has no read access.

    Args:
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        None
    """
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    (root_dir / ".marimba").mkdir()

    os.chmod(root_dir, 0o000)  # Remove read access
    try:
        with pytest.raises(typer.Exit):
            find_project_dir_or_exit(root_dir)
    finally:
        os.chmod(root_dir, 0o755)  # Restore permissions for cleanup


@pytest.mark.integration
def test_find_project_dir_or_exit_with_marimba_as_file(tmp_path: Path) -> None:
    """
    Test find_project_dir_or_exit when .marimba is a file, not a directory.

    Args:
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        None
    """
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    (root_dir / ".marimba").touch()  # Create .marimba as a file

    with pytest.raises(typer.Exit):
        find_project_dir_or_exit(root_dir)


# ---------------------------------------------------------------------------------------------------------------------#
# Testing project()
# ---------------------------------------------------------------------------------------------------------------------#


@pytest.fixture
def setup_test_directory(tmp_path: Path) -> Path:
    """
    Fixture to set up a temporary directory for testing.

    Args:
        tmp_path: Temporary directory path provided by pytest.

    Returns:
        The path to the temporary directory.
    """
    return tmp_path


@pytest.mark.integration
def test_project_creates_new_project(setup_test_directory: Path, test_data_factory: TestDataFactory) -> None:
    """
    Test project creates a new Marimba project successfully with real functionality.

    Args:
        setup_test_directory: Path to the temporary directory.
        test_data_factory: Factory for creating test data.

    Returns:
        None
    """
    project_dir = setup_test_directory / "new_project"

    # Test real project creation without excessive mocking
    result = runner.invoke(marimba_cli, ["new", "project", str(project_dir)])
    print(result.output)
    assert result.exit_code == 0
    assert "Created new Marimba project at" in result.output

    # Verify real project structure was created using helper function
    assert_project_structure_exists(project_dir, "New project")


@pytest.mark.integration
def test_project_exits_if_project_exists(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test project exits with an error if the project directory already exists.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "existing_project"
    project_dir.mkdir()

    mock_create = mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create",
        side_effect=FileExistsError,
    )
    result = runner.invoke(marimba_cli, ["new", "project", str(project_dir)])
    assert result.exit_code != 0
    mock_create.assert_called_once_with(project_dir)
    assert "A Marimba project already exists at:" in result.output


@pytest.mark.integration
def test_project_logs_command_execution(setup_test_directory: Path) -> None:
    """
    Test project logs the command execution.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "log_test_project"

    result = runner.invoke(marimba_cli, ["new", "project", str(project_dir)])
    assert result.exit_code == 0
    assert "Created new Marimba project at" in result.output


# def test_project_handles_invalid_path(tmp_path: Path) -> None:
#     """
#     Test project handles invalid paths gracefully.
#
#     Args:
#         tmp_path: Temporary directory path provided by pytest.
#
#     Returns:
#         None
#     """
#     invalid_path = tmp_path / "invalid_path"
#
#     with patch("marimba.core.wrappers.project.ProjectWrapper.create", side_effect=ValueError("Invalid path")):
#         result = runner.invoke(marimba, ["new", "project", str(invalid_path)])
#         assert result.exit_code != 0
#         assert "Invalid path" in result.output


@pytest.mark.integration
def test_project_prints_success_message(setup_test_directory: Path) -> None:
    """
    Test project prints a success message upon creating a new project with real functionality.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "success_message_project"

    # Test real project creation and verify success message
    result = runner.invoke(marimba_cli, ["new", "project", str(project_dir)])
    assert result.exit_code == 0
    assert "Created new Marimba project at:" in result.output

    # Also verify the project was actually created
    assert project_dir.exists(), "Project should actually be created"


@pytest.mark.integration
def test_project_exit_code_on_success(setup_test_directory: Path) -> None:
    """
    Test project exit code is zero upon successful project creation with real functionality.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "exit_code_success_project"

    # Test real project creation and verify exit code
    result = runner.invoke(marimba_cli, ["new", "project", str(project_dir)])
    assert result.exit_code == 0

    # Verify the project was actually created (not just mocked)
    assert project_dir.exists() and (project_dir / ".marimba").exists()


@pytest.mark.integration
def test_project_exit_code_on_failure(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test project exit code is non-zero upon project creation failure.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "exit_code_failure_project"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create",
        side_effect=FileExistsError,
    )
    result = runner.invoke(marimba_cli, ["new", "project", str(project_dir)])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------------------------------------------------#
# Testing pipeline()
# ---------------------------------------------------------------------------------------------------------------------#


@pytest.mark.integration
def test_pipeline_creates_new_pipeline(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test pipeline command creates a new pipeline successfully.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    pipeline_name = "test_pipeline"
    url = "https://example.com/repo.git"

    mock_create_pipeline = mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_pipeline",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create",
        return_value=mocker.MagicMock(),
    )
    result = runner.invoke(
        marimba_cli,
        ["new", "pipeline", pipeline_name, url, "--project-dir", str(project_dir)],
    )
    assert result.exit_code == 0
    mock_create_pipeline.assert_called_once_with(pipeline_name, url, {})
    assert "Created new Marimba pipeline" in result.output


@pytest.mark.integration
def test_pipeline_invalid_name_error(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test pipeline command exits with an error for an invalid pipeline name.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    pipeline_name = "invalid/name"
    url = "https://example.com/repo.git"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_pipeline",
        side_effect=ProjectWrapper.InvalidNameError("Invalid name"),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    result = runner.invoke(
        marimba_cli,
        ["new", "pipeline", pipeline_name, url, "--project-dir", str(project_dir)],
    )
    assert result.exit_code != 0
    assert "Invalid pipeline name:" in result.output


@pytest.mark.integration
def test_pipeline_creation_failure(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test pipeline command exits with an error if pipeline creation fails.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    pipeline_name = "test_pipeline"
    url = "https://example.com/repo.git"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_pipeline",
        side_effect=Exception("Creation failed"),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    result = runner.invoke(
        marimba_cli,
        ["new", "pipeline", pipeline_name, url, "--project-dir", str(project_dir)],
    )
    assert result.exit_code != 0
    assert "Could not create pipeline:" in result.output


@pytest.mark.integration
def test_pipeline_logs_command_execution(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test pipeline command logs the command execution.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    pipeline_name = "test_pipeline"
    url = "https://example.com/repo.git"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_pipeline",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    result = runner.invoke(
        marimba_cli,
        ["new", "pipeline", pipeline_name, url, "--project-dir", str(project_dir)],
    )
    assert result.exit_code == 0
    assert "Created new Marimba pipeline" in result.output


# ---------------------------------------------------------------------------------------------------------------------#
# Testing collection()
# ---------------------------------------------------------------------------------------------------------------------#


@pytest.mark.integration
def test_collection_creates_new_collection(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test collection command creates a new collection successfully.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    collection_name = "test_collection"
    parent_collection_name = "parent_collection"

    mock_create_collection = mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_collection",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.prompt_collection_config",
        return_value=mocker.MagicMock(),
    )
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
    assert result.exit_code == 0
    mock_create_collection.assert_called_once_with(collection_name, mocker.ANY)


@pytest.mark.integration
def test_collection_invalid_name_error(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test collection command exits with an error for an invalid collection name.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    collection_name = "invalid/name"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_collection",
        side_effect=ProjectWrapper.InvalidNameError("Invalid name"),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    result = runner.invoke(
        marimba_cli,
        ["new", "collection", collection_name, "--project-dir", str(project_dir)],
    )
    assert result.exit_code != 0
    assert "Invalid collection name:" in result.output


@pytest.mark.integration
def test_collection_no_such_parent_collection_error(
    mocker: pytest_mock.MockerFixture, setup_test_directory: Path
) -> None:
    """
    Test collection command exits with an error if the specified parent collection does not exist.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    collection_name = "test_collection"
    parent_collection_name = "non_existent_parent"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_collection",
        side_effect=ProjectWrapper.NoSuchCollectionError("No such parent collection"),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
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
    assert result.exit_code != 0
    assert "No such parent collection:" in result.output


@pytest.mark.integration
def test_collection_creation_failure(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test collection command exits with an error if collection creation fails.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    collection_name = "test_collection"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_collection",
        side_effect=ProjectWrapper.CreateCollectionError("Creation failed"),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    result = runner.invoke(
        marimba_cli,
        ["new", "collection", collection_name, "--project-dir", str(project_dir)],
    )
    assert result.exit_code != 0
    assert "Could not create collection:" in result.output


@pytest.mark.integration
def test_collection_creation_other_failure(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test collection command exits with an error if collection creation fails.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    collection_name = "test_collection"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_collection",
        side_effect=Exception("Creation failed"),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    result = runner.invoke(
        marimba_cli,
        ["new", "collection", collection_name, "--project-dir", str(project_dir)],
    )
    assert result.exit_code != 0
    assert "Could not create collection:" in result.output


@pytest.mark.integration
def test_collection_logs_command_execution(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test collection command logs the command execution.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    collection_name = "test_collection"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_collection",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    result = runner.invoke(
        marimba_cli,
        ["new", "collection", collection_name, "--project-dir", str(project_dir)],
    )
    assert result.exit_code == 0
    assert "Created new Marimba collection" in result.output


# ---------------------------------------------------------------------------------------------------------------------#
# Testing target()
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.integration
def test_target_creates_new_target(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test target command creates a new distribution target successfully.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    target_name = "test_target"

    mock_create_target = mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_target",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    mocker.patch(
        "marimba.core.wrappers.project.DistributionTargetWrapper.prompt_target",
        return_value=("target_type", "target_config"),
    )
    result = runner.invoke(
        marimba_cli,
        ["new", "target", target_name, "--project-dir", str(project_dir)],
    )
    assert result.exit_code == 0
    mock_create_target.assert_called_once_with(target_name, "target_type", "target_config")
    assert "Created new Marimba target" in result.output


@pytest.mark.integration
def test_target_invalid_name_error(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test target command exits with an error for an invalid target name.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    target_name = "invalid/name"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_target",
        side_effect=ProjectWrapper.InvalidNameError("Invalid name"),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    mocker.patch(
        "marimba.core.wrappers.project.DistributionTargetWrapper.prompt_target",
        return_value=("target_type", "target_config"),
    )
    result = runner.invoke(
        marimba_cli,
        ["new", "target", target_name, "--project-dir", str(project_dir)],
    )
    assert result.exit_code != 0
    assert "Invalid target name:" in result.output


@pytest.mark.integration
def test_target_already_exists_error(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test target command exits with an error if the target already exists.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    target_name = "existing_target"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_target",
        side_effect=FileExistsError,
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    mocker.patch(
        "marimba.core.wrappers.project.DistributionTargetWrapper.prompt_target",
        return_value=("target_type", "target_config"),
    )
    result = runner.invoke(
        marimba_cli,
        ["new", "target", target_name, "--project-dir", str(project_dir)],
    )
    print(result.output)
    assert result.exit_code != 0
    assert "A Marimba target already exists" in result.output


@pytest.mark.integration
def test_target_creation_failure(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test target command exits with an error if target creation fails.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    target_name = "test_target"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_target",
        side_effect=Exception("Creation failed"),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    result = runner.invoke(
        marimba_cli,
        ["new", "target", target_name, "--project-dir", str(project_dir)],
    )
    assert result.exit_code != 0
    assert "Could not create target:" in result.output


@pytest.mark.integration
def test_target_logs_command_execution(mocker: pytest_mock.MockerFixture, setup_test_directory: Path) -> None:
    """
    Test target command logs the command execution.

    Args:
        setup_test_directory: Path to the temporary directory.

    Returns:
        None
    """
    project_dir = setup_test_directory / "project"
    project_dir.mkdir()
    target_name = "test_target"

    mocker.patch(
        "marimba.core.wrappers.project.ProjectWrapper.create_target",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("marimba.core.cli.new.find_project_dir_or_exit", return_value=project_dir)
    mocker.patch(
        "marimba.core.wrappers.project.DistributionTargetWrapper.prompt_target",
        return_value=("target_type", "target_config"),
    )
    result = runner.invoke(
        marimba_cli,
        ["new", "target", target_name, "--project-dir", str(project_dir)],
    )
    assert result.exit_code == 0
    assert "Created new Marimba target" in result.output
