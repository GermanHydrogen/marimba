"""Tests for marimba.lib.exif module."""

from pathlib import Path

import pytest
from exiftool.exceptions import ExifToolException

from marimba.lib.exif import get_dict


class TestExifUtilities:
    """Test EXIF utility functions."""

    @pytest.fixture
    def test_image_path(self, tmp_path):
        """Create a test image path."""
        return tmp_path / "test_image.jpg"

    @pytest.mark.integration
    def test_get_dict_with_metadata(self, mocker, test_image_path):
        """Test getting EXIF data from image with metadata."""
        # Mock the ExifToolHelper context manager and metadata
        mock_exiftool_helper = mocker.patch("exiftool.ExifToolHelper")
        mock_et = mocker.Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        expected_metadata = {"FileName": "test_image.jpg", "ImageWidth": 800}
        mock_et.get_metadata.return_value = [expected_metadata]

        result = get_dict(test_image_path)

        assert result == expected_metadata
        mock_et.get_metadata.assert_called_once_with(str(test_image_path))

    @pytest.mark.integration
    def test_get_dict_no_metadata(self, mocker, test_image_path):
        """Test getting EXIF data from image without metadata."""
        mock_exiftool_helper = mocker.patch("exiftool.ExifToolHelper")
        mock_et = mocker.Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        mock_et.get_metadata.return_value = []  # No metadata

        result = get_dict(test_image_path)

        assert result is None
        mock_et.get_metadata.assert_called_once_with(str(test_image_path))

    @pytest.mark.integration
    def test_get_dict_empty_metadata(self, mocker, test_image_path):
        """Test getting EXIF data when metadata is None."""
        mock_exiftool_helper = mocker.patch("exiftool.ExifToolHelper")
        mock_et = mocker.Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        mock_et.get_metadata.return_value = None  # None metadata

        result = get_dict(test_image_path)

        assert result is None
        mock_et.get_metadata.assert_called_once_with(str(test_image_path))

    @pytest.mark.integration
    def test_get_dict_with_string_path(self, mocker):
        """Test getting EXIF data using string path."""
        mock_exiftool_helper = mocker.patch("exiftool.ExifToolHelper")
        mock_et = mocker.Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        expected_metadata = {"FileName": "test_image.jpg"}
        mock_et.get_metadata.return_value = [expected_metadata]

        string_path = "/path/to/image.jpg"
        result = get_dict(string_path)

        assert result == expected_metadata
        mock_et.get_metadata.assert_called_once_with(string_path)

    @pytest.mark.integration
    def test_get_dict_exiftool_not_found(self, mocker, test_image_path):
        """Test handling when exiftool is not found."""
        mock_exiftool_helper = mocker.patch("exiftool.ExifToolHelper")
        mock_show_error = mocker.patch("marimba.lib.exif.show_dependency_error_and_exit")
        mock_exiftool_helper.side_effect = FileNotFoundError("exiftool not found")

        result = get_dict(test_image_path)

        assert result is None
        mock_show_error.assert_called_once()

    @pytest.mark.integration
    def test_get_dict_file_not_found_other(self, mocker, test_image_path):
        """Test handling FileNotFoundError not related to exiftool."""
        mock_exiftool_helper = mocker.patch("exiftool.ExifToolHelper")
        mock_exiftool_helper.side_effect = FileNotFoundError("Image file not found")

        result = get_dict(test_image_path)

        assert result is None

    @pytest.mark.integration
    def test_get_dict_exiftool_exception(self, mocker, test_image_path):
        """Test handling ExifToolException."""
        mock_exiftool_helper = mocker.patch("exiftool.ExifToolHelper")
        mock_exiftool_helper.side_effect = ExifToolException("Invalid EXIF data")

        result = get_dict(test_image_path)

        assert result is None

    @pytest.mark.integration
    def test_get_dict_multiple_images(self, mocker, test_image_path):
        """Test getting EXIF data when multiple items in metadata list."""
        mock_exiftool_helper = mocker.patch("exiftool.ExifToolHelper")
        mock_et = mocker.Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        # Multiple metadata entries - should return first one
        metadata_list = [
            {"FileName": "image1.jpg", "ImageWidth": 800},
            {"FileName": "image2.jpg", "ImageWidth": 600},
        ]
        mock_et.get_metadata.return_value = metadata_list

        result = get_dict(test_image_path)

        assert result == metadata_list[0]
        mock_et.get_metadata.assert_called_once_with(str(test_image_path))

    @pytest.mark.integration
    def test_get_dict_various_metadata_types(self, mocker, test_image_path):
        """Test getting EXIF data with various metadata field types."""
        mock_exiftool_helper = mocker.patch("exiftool.ExifToolHelper")
        mock_et = mocker.Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        complex_metadata = {
            "FileName": "test_image.jpg",
            "ImageWidth": 800,
            "ImageHeight": 600,
            "Make": "Canon",
            "Model": "EOS R5",
            "DateTime": "2023:12:25 10:30:00",
            "GPS:GPSLatitude": 37.7749,
            "GPS:GPSLongitude": -122.4194,
            "EXIF:ExposureTime": "1/60",
            "EXIF:FNumber": 5.6,
        }
        mock_et.get_metadata.return_value = [complex_metadata]

        result = get_dict(test_image_path)

        assert result == complex_metadata
        assert result["FileName"] == "test_image.jpg"
        assert result["ImageWidth"] == 800
        assert result["GPS:GPSLatitude"] == 37.7749
