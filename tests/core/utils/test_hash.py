"""Tests for marimba.core.utils.hash module."""

import hashlib
from pathlib import Path

import pytest

from marimba.core.utils.hash import compute_hash


class TestHashUtilities:
    """Test hash utility functions."""

    @pytest.fixture
    def test_file(self, tmp_path):
        """Create a test file with known content."""
        test_file = tmp_path / "test_file.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)
        return test_file, test_content

    @pytest.fixture
    def test_directory(self, tmp_path):
        """Create a test directory."""
        test_dir = tmp_path / "test_directory"
        test_dir.mkdir()
        return test_dir

    @pytest.fixture
    def test_root_dir(self, tmp_path):
        """Create a root directory for relative path testing."""
        root_dir = tmp_path / "root"
        root_dir.mkdir()
        return root_dir

    @pytest.mark.integration
    def test_compute_hash_file_contents(self, test_file):
        """Test computing hash of file contents."""
        file_path, content = test_file

        # Calculate expected hash
        expected_hash = hashlib.sha256(content).hexdigest()

        result = compute_hash(file_path)

        assert result == expected_hash

    @pytest.mark.integration
    def test_compute_hash_large_file(self, tmp_path):
        """Test computing hash of large file (multiple chunks)."""
        large_file = tmp_path / "large_file.txt"

        # Create content larger than chunk size to test chunking (reduced from 1MB for performance)
        chunk_size = 64_000  # 64KB - sufficient to test chunking behavior
        content = b"A" * (chunk_size + 1000)  # Slightly over 64KB
        large_file.write_bytes(content)

        expected_hash = hashlib.sha256(content).hexdigest()

        result = compute_hash(large_file)

        assert result == expected_hash

    @pytest.mark.integration
    def test_compute_hash_empty_file(self, tmp_path):
        """Test computing hash of empty file."""
        empty_file = tmp_path / "empty_file.txt"
        empty_file.touch()

        expected_hash = hashlib.sha256().hexdigest()

        result = compute_hash(empty_file)

        assert result == expected_hash

    @pytest.mark.integration
    def test_compute_hash_directory_absolute_path(self, test_directory):
        """Test computing hash of directory using absolute path."""
        expected_hash = hashlib.sha256(str(test_directory.as_posix()).encode()).hexdigest()

        result = compute_hash(test_directory)

        assert result == expected_hash

    @pytest.mark.integration
    def test_compute_hash_directory_with_root_dir(self, test_root_dir):
        """Test computing hash of directory with root directory."""
        test_dir = test_root_dir / "subdir"
        test_dir.mkdir()

        # Expected hash should be of the relative path
        relative_path = test_dir.relative_to(test_root_dir)
        expected_hash = hashlib.sha256(str(relative_path.as_posix()).encode()).hexdigest()

        result = compute_hash(test_dir, root_dir=test_root_dir)

        assert result == expected_hash

    @pytest.mark.integration
    def test_compute_hash_file_with_root_dir_ignored(self, test_root_dir):
        """Test that root_dir is ignored for files."""
        test_file = test_root_dir / "test.txt"
        content = b"File content"
        test_file.write_bytes(content)

        expected_hash = hashlib.sha256(content).hexdigest()

        result = compute_hash(test_file, root_dir=test_root_dir)

        assert result == expected_hash

    @pytest.mark.integration
    def test_compute_hash_directory_outside_root(self, tmp_path, test_root_dir):
        """Test error when directory is outside root directory."""
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        with pytest.raises(ValueError, match="is not within root directory"):
            compute_hash(outside_dir, root_dir=test_root_dir)

    @pytest.mark.integration
    def test_compute_hash_nested_directory_with_root(self, test_root_dir):
        """Test computing hash of deeply nested directory with root."""
        nested_dir = test_root_dir / "level1" / "level2" / "level3"
        nested_dir.mkdir(parents=True)

        relative_path = nested_dir.relative_to(test_root_dir)
        expected_hash = hashlib.sha256(str(relative_path.as_posix()).encode()).hexdigest()

        result = compute_hash(nested_dir, root_dir=test_root_dir)

        assert result == expected_hash

    @pytest.mark.integration
    def test_compute_hash_symlink_as_directory(self, tmp_path):
        """Test computing hash of symlink to directory."""
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        symlink = tmp_path / "symlink"
        symlink.symlink_to(target_dir)

        # Should hash the symlink path itself (since it's not a file)
        expected_hash = hashlib.sha256(str(symlink.as_posix()).encode()).hexdigest()

        result = compute_hash(symlink)

        assert result == expected_hash

    @pytest.mark.integration
    def test_compute_hash_nonexistent_path_as_directory(self, tmp_path):
        """Test computing hash of non-existent path (treated as directory)."""
        nonexistent = tmp_path / "nonexistent"

        expected_hash = hashlib.sha256(str(nonexistent.as_posix()).encode()).hexdigest()

        result = compute_hash(nonexistent)

        assert result == expected_hash

    @pytest.mark.integration
    def test_compute_hash_file_read_error(self, mocker, tmp_path):
        """Test handling of file read error."""
        test_file = tmp_path / "test_file.txt"
        test_file.touch()  # Create the file so is_file() returns True

        # Mock the open method to raise OSError
        mock_open_method = mocker.patch("pathlib.Path.open")
        mock_open_method.side_effect = OSError("Permission denied")

        with pytest.raises(OSError, match="Failed to read file"):
            compute_hash(test_file)

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "content1,content2,should_be_equal,description",
        [
            ("Content 1", "Content 2", False, "different content produces different hashes"),
            ("Same content", "Same content", True, "same content produces same hash"),
            ("", "", True, "empty files produce same hash"),
            ("A" * 1000, "B" * 1000, False, "different large content produces different hashes"),
        ],
    )
    def test_compute_hash_file_content_comparison(self, tmp_path, content1, content2, should_be_equal, description):
        """Test file hash comparison with various content scenarios."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text(content1)
        file2.write_text(content2)

        hash1 = compute_hash(file1)
        hash2 = compute_hash(file2)

        if should_be_equal:
            assert hash1 == hash2, f"Hashes should be equal for {description}"
        else:
            assert hash1 != hash2, f"Hashes should be different for {description}"

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "dir1_name,dir2_name,description",
        [
            ("dir1", "dir2", "different simple directory names"),
            ("special-dir_with@symbols", "another-special#dir", "special characters in names"),
            ("nested/path/dir1", "nested/path/dir2", "nested directory paths"),
            ("测试目录1", "测试目录2", "unicode directory names"),
        ],
    )
    def test_compute_hash_directory_different_paths_different_hashes(self, tmp_path, dir1_name, dir2_name, description):
        """Test that different directory paths produce different hashes."""
        dir1 = tmp_path / dir1_name
        dir2 = tmp_path / dir2_name

        dir1.mkdir(parents=True)
        dir2.mkdir(parents=True)

        hash1 = compute_hash(dir1)
        hash2 = compute_hash(dir2)

        assert hash1 != hash2, f"Different hashes expected for {description}"

    @pytest.mark.integration
    def test_compute_hash_relative_vs_absolute_paths_with_root(self, test_root_dir):
        """Test that relative path calculation handles absolute vs relative inputs."""
        subdir = test_root_dir / "subdir"
        subdir.mkdir()

        # Test with absolute input
        absolute_result = compute_hash(subdir.absolute(), root_dir=test_root_dir)

        # Test with another absolute input (should be same)
        absolute_result2 = compute_hash(subdir, root_dir=test_root_dir)

        # Both should produce the same hash
        assert absolute_result == absolute_result2

    @pytest.mark.integration
    def test_compute_hash_special_characters_in_path(self, tmp_path):
        """Test computing hash of path with special characters."""
        special_dir = tmp_path / "special-dir_with@symbols"
        special_dir.mkdir()

        expected_hash = hashlib.sha256(str(special_dir.as_posix()).encode()).hexdigest()

        result = compute_hash(special_dir)

        assert result == expected_hash

    @pytest.mark.integration
    def test_compute_hash_unicode_path(self, tmp_path):
        """Test computing hash of path with Unicode characters."""
        unicode_dir = tmp_path / "测试目录"
        unicode_dir.mkdir()

        expected_hash = hashlib.sha256(str(unicode_dir.as_posix()).encode()).hexdigest()

        result = compute_hash(unicode_dir)

        assert result == expected_hash

    @pytest.mark.integration
    def test_compute_hash_returns_lowercase_hex(self, test_file):
        """Test that hash is returned as lowercase hexadecimal."""
        file_path, _ = test_file

        result = compute_hash(file_path)

        # Check it's a valid hex string
        assert len(result) == 64  # SHA-256 produces 64-character hex
        assert all(c in "0123456789abcdef" for c in result)
        assert result.islower()

    @pytest.mark.integration
    def test_compute_hash_consistency(self, test_file):
        """Test that computing hash multiple times gives same result."""
        file_path, _ = test_file

        hash1 = compute_hash(file_path)
        hash2 = compute_hash(file_path)
        hash3 = compute_hash(file_path)

        assert hash1 == hash2 == hash3

    @pytest.mark.integration
    def test_compute_hash_directory_resolve_error_with_root(self, mocker, tmp_path, test_root_dir):
        """Test error handling when path.resolve() fails during relative path calculation."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        # Make the path's resolve() method raise ValueError
        mock_resolve = mocker.patch("pathlib.Path.resolve")
        mock_resolve.side_effect = ValueError("Path resolution error")

        with pytest.raises(ValueError, match="is not within root directory"):
            compute_hash(test_dir, root_dir=test_root_dir)
