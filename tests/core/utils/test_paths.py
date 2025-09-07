"""Tests for marimba.core.utils.paths module."""

from pathlib import Path

import pytest
import typer

from marimba.core.utils.paths import (
    detect_hardlinked_files,
    detect_readonly_files,
    find_project_dir,
    find_project_dir_or_exit,
    format_path_for_logging,
    hardlink_path,
    remove_directory_tree,
)


class TestFindProjectDir:
    """Test find_project_dir function."""

    @pytest.fixture
    def temp_project_structure(self, tmp_path):
        """Create temporary project structure."""
        # Create project structure
        project_root = tmp_path / "project_root"
        project_root.mkdir()
        (project_root / ".marimba").mkdir()

        # Create subdirectory
        subdir = project_root / "subdir"
        subdir.mkdir()

        return project_root, subdir

    @pytest.mark.unit
    def test_find_project_dir_from_root(self, temp_project_structure):
        """Test finding project directory from root."""
        project_root, _ = temp_project_structure

        result = find_project_dir(project_root)

        assert result == project_root

    @pytest.mark.unit
    def test_find_project_dir_from_subdirectory(self, temp_project_structure):
        """Test finding project directory from subdirectory."""
        project_root, subdir = temp_project_structure

        result = find_project_dir(subdir)

        assert result == project_root

    @pytest.mark.unit
    def test_find_project_dir_string_path(self, temp_project_structure):
        """Test finding project directory with string path."""
        project_root, subdir = temp_project_structure

        result = find_project_dir(str(subdir))

        assert result == project_root

    @pytest.mark.unit
    def test_find_project_dir_not_found(self, tmp_path):
        """Test finding project directory when not found."""
        non_project = tmp_path / "not_a_project"
        non_project.mkdir()

        result = find_project_dir(non_project)

        assert result is None

    @pytest.mark.unit
    def test_find_project_dir_no_read_access(self, mocker, tmp_path):
        """Test finding project directory with no read access."""
        mock_access = mocker.patch("os.access")
        mock_access.return_value = False

        result = find_project_dir(tmp_path)

        assert result is None

    @pytest.mark.unit
    def test_find_project_dir_marimba_is_file(self, tmp_path):
        """Test finding project directory when .marimba is a file, not directory."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".marimba").touch()  # Create as file, not directory

        result = find_project_dir(project_dir)

        assert result is None


class TestFindProjectDirOrExit:
    """Test find_project_dir_or_exit function."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create temporary project."""
        project_root = tmp_path / "test_project"
        project_root.mkdir()
        (project_root / ".marimba").mkdir()
        return project_root

    @pytest.mark.unit
    def test_find_project_dir_or_exit_with_valid_dir(self, temp_project):
        """Test find_project_dir_or_exit with valid project directory."""
        result = find_project_dir_or_exit(temp_project)

        assert result == temp_project

    @pytest.mark.unit
    def test_find_project_dir_or_exit_with_none_uses_cwd(self, mocker, tmp_path):
        """Test find_project_dir_or_exit with None uses current working directory."""
        mock_cwd = mocker.patch("marimba.core.utils.paths.Path.cwd")
        mock_find = mocker.patch("marimba.core.utils.paths.find_project_dir")
        mock_cwd.return_value = tmp_path
        mock_find.return_value = tmp_path

        result = find_project_dir_or_exit(None)

        assert result == tmp_path
        mock_find.assert_called_once_with(tmp_path)

    @pytest.mark.unit
    def test_find_project_dir_or_exit_not_found_exits(self, mocker, tmp_path):
        """Test find_project_dir_or_exit exits when project not found."""
        mock_find = mocker.patch("marimba.core.utils.paths.find_project_dir")
        mock_find.return_value = None

        with pytest.raises(typer.Exit) as exc_info:
            find_project_dir_or_exit(tmp_path)

        assert exc_info.value.exit_code == 1


class TestRemoveDirectoryTree:
    """Test remove_directory_tree function."""

    @pytest.mark.unit
    def test_remove_directory_tree_dry_run(self, tmp_path):
        """Test remove_directory_tree with dry_run=True."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        (test_dir / "test_file.txt").touch()

        # Should not raise error and directory should still exist
        remove_directory_tree(test_dir, "test entity", dry_run=True)

        assert test_dir.exists()

    @pytest.mark.unit
    def test_remove_directory_tree_actual_deletion(self, mocker, tmp_path):
        """Test remove_directory_tree with actual deletion."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        mock_rmtree = mocker.patch("shutil.rmtree")

        remove_directory_tree(test_dir, "test entity", dry_run=False)

        mock_rmtree.assert_called_once_with(test_dir)

    @pytest.mark.unit
    def test_remove_directory_tree_invalid_directory(self, tmp_path):
        """Test remove_directory_tree with invalid directory."""
        non_existent = tmp_path / "non_existent"

        with pytest.raises(typer.Exit) as exc_info:
            remove_directory_tree(non_existent, "test entity", dry_run=False)

        assert exc_info.value.exit_code == 1

    @pytest.mark.unit
    def test_remove_directory_tree_deletion_error(self, mocker, tmp_path):
        """Test remove_directory_tree with deletion error."""
        mock_rmtree = mocker.patch("shutil.rmtree")
        mock_rmtree.side_effect = OSError("Permission denied")
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        with pytest.raises(typer.Exit) as exc_info:
            remove_directory_tree(test_dir, "test entity", dry_run=False)

        assert exc_info.value.exit_code == 1


class TestHardlinkPath:
    """Test hardlink_path function."""

    @pytest.mark.unit
    def test_hardlink_path_invalid_source(self, tmp_path):
        """Test hardlink_path with invalid source directory."""
        non_existent = tmp_path / "non_existent"
        dest = tmp_path / "dest"

        with pytest.raises(typer.Exit) as exc_info:
            hardlink_path(non_existent, dest, dry_run=False)

        assert exc_info.value.exit_code == 1

    @pytest.mark.unit
    def test_hardlink_path_source_is_file(self, tmp_path):
        """Test hardlink_path when source is a file instead of directory."""
        source_file = tmp_path / "source_file.txt"
        source_file.touch()
        dest = tmp_path / "dest"

        with pytest.raises(typer.Exit) as exc_info:
            hardlink_path(source_file, dest, dry_run=False)

        assert exc_info.value.exit_code == 1

    @pytest.mark.unit
    def test_hardlink_path_dry_run(self, tmp_path):
        """Test hardlink_path with dry_run=True."""
        # Create source structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "file1.txt").touch()
        subdir = src_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").touch()

        dest_dir = tmp_path / "dest"

        hardlink_path(src_dir, dest_dir, dry_run=True)

        # Destination should be created but no hard links actually made
        assert dest_dir.exists()

    @pytest.mark.unit
    def test_hardlink_path_actual_linking(self, tmp_path):
        """Test hardlink_path with actual hard link creation."""
        # Create source structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        src_file = src_dir / "file1.txt"
        src_file.write_text("test content")

        dest_dir = tmp_path / "dest"

        hardlink_path(src_dir, dest_dir, dry_run=False)

        # Check that hard link was created
        dest_file = dest_dir / "file1.txt"
        assert dest_file.exists()
        assert dest_file.read_text() == "test content"

    @pytest.mark.unit
    def test_hardlink_path_linking_error(self, mocker, tmp_path):
        """Test hardlink_path with hard link creation error."""
        # Create source structure
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        src_file = src_dir / "file1.txt"
        src_file.touch()

        dest_dir = tmp_path / "dest"

        mocker.patch.object(Path, "hardlink_to", side_effect=OSError("Hard link failed"))
        # Should not raise exception, just log error
        hardlink_path(src_dir, dest_dir, dry_run=False)


class TestDetectHardlinkedFiles:
    """Test detect_hardlinked_files function."""

    @pytest.mark.unit
    def test_detect_hardlinked_files_empty_list(self):
        """Test detect_hardlinked_files with empty list."""
        result = detect_hardlinked_files([])
        assert result == []

    @pytest.mark.unit
    def test_detect_hardlinked_files_non_existent_file(self, tmp_path):
        """Test detect_hardlinked_files with non-existent file."""
        non_existent = tmp_path / "non_existent.txt"

        result = detect_hardlinked_files([non_existent])

        assert result == []

    @pytest.mark.unit
    def test_detect_hardlinked_files_directory(self, tmp_path):
        """Test detect_hardlinked_files with directory instead of file."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        result = detect_hardlinked_files([test_dir])

        assert result == []

    @pytest.mark.unit
    def test_detect_hardlinked_files_single_link(self, tmp_path):
        """Test detect_hardlinked_files with file having single link."""
        test_file = tmp_path / "test_file.txt"
        test_file.touch()

        result = detect_hardlinked_files([test_file])

        # Single link should not be detected as hardlinked
        assert result == []

    @pytest.mark.unit
    def test_detect_hardlinked_files_multiple_links(self, tmp_path):
        """Test detect_hardlinked_files with file having multiple hard links."""
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("test content")

        # Create hard link
        hard_link = tmp_path / "hard_link.txt"
        hard_link.hardlink_to(test_file)

        result = detect_hardlinked_files([test_file, hard_link])

        # Both should be detected as hardlinked
        assert len(result) == 2
        assert test_file in result
        assert hard_link in result

    @pytest.mark.unit
    def test_detect_hardlinked_files_stat_error(self, tmp_path):
        """Test detect_hardlinked_files with stat error."""
        test_file = tmp_path / "test_file.txt"
        test_file.touch()

        # Remove the file to cause a stat error
        test_file.unlink()

        result = detect_hardlinked_files([test_file])

        # Should handle error gracefully
        assert result == []


class TestDetectReadonlyFiles:
    """Test detect_readonly_files function."""

    @pytest.mark.unit
    def test_detect_readonly_files_empty_list(self):
        """Test detect_readonly_files with empty list."""
        result = detect_readonly_files([])
        assert result == []

    @pytest.mark.unit
    def test_detect_readonly_files_non_existent_file(self, tmp_path):
        """Test detect_readonly_files with non-existent file."""
        non_existent = tmp_path / "non_existent.txt"

        result = detect_readonly_files([non_existent])

        assert result == []

    @pytest.mark.unit
    def test_detect_readonly_files_directory(self, tmp_path):
        """Test detect_readonly_files with directory instead of file."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        result = detect_readonly_files([test_dir])

        assert result == []

    @pytest.mark.unit
    def test_detect_readonly_files_writable_file(self, mocker, tmp_path):
        """Test detect_readonly_files with writable file."""
        mock_access = mocker.patch("os.access")
        mock_access.return_value = True
        test_file = tmp_path / "test_file.txt"
        test_file.touch()

        result = detect_readonly_files([test_file])

        assert result == []

    @pytest.mark.unit
    def test_detect_readonly_files_readonly_file(self, mocker, tmp_path):
        """Test detect_readonly_files with read-only file."""
        mock_access = mocker.patch("os.access")
        mock_access.return_value = False
        test_file = tmp_path / "test_file.txt"
        test_file.touch()

        result = detect_readonly_files([test_file])

        assert test_file in result

    @pytest.mark.unit
    def test_detect_readonly_files_access_error(self, mocker, tmp_path):
        """Test detect_readonly_files with access permission error."""
        mock_access = mocker.patch("os.access")
        mock_access.side_effect = PermissionError("Access denied")
        test_file = tmp_path / "test_file.txt"
        test_file.touch()

        result = detect_readonly_files([test_file])

        # Should assume not writable on error
        assert test_file in result


class TestFormatPathForLogging:
    """Test format_path_for_logging function."""

    @pytest.mark.unit
    def test_format_path_for_logging_with_project_dir(self, tmp_path):
        """Test format_path_for_logging with provided project directory."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        test_path = project_dir / "subdir" / "file.txt"

        result = format_path_for_logging(test_path, project_dir)

        assert result == "subdir/file.txt"

    @pytest.mark.unit
    def test_format_path_for_logging_string_path(self, tmp_path):
        """Test format_path_for_logging with string path."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        test_path = project_dir / "file.txt"

        result = format_path_for_logging(str(test_path), project_dir)

        assert result == "file.txt"

    @pytest.mark.unit
    def test_format_path_for_logging_without_project_dir_found(self, mocker, tmp_path):
        """Test format_path_for_logging without project_dir, but project found."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        test_path = project_dir / "file.txt"
        mock_find = mocker.patch("marimba.core.utils.paths.find_project_dir")
        mock_find.return_value = project_dir

        result = format_path_for_logging(test_path)

        assert result == "file.txt"

    @pytest.mark.unit
    def test_format_path_for_logging_without_project_dir_not_found(self, mocker, tmp_path):
        """Test format_path_for_logging without project_dir and project not found."""
        mock_find = mocker.patch("marimba.core.utils.paths.find_project_dir")
        mock_find.return_value = None
        test_path = tmp_path / "file.txt"

        result = format_path_for_logging(test_path)

        assert result == str(test_path.resolve())

    @pytest.mark.unit
    def test_format_path_for_logging_path_outside_project(self, tmp_path):
        """Test format_path_for_logging with path outside project directory."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        outside_path = tmp_path / "outside" / "file.txt"

        result = format_path_for_logging(outside_path, project_dir)

        # Should return absolute path when cannot be made relative
        assert result == str(outside_path.resolve())
