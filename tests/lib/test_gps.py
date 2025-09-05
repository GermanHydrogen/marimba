"""Tests for marimba.lib.gps module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from marimba.lib.gps import (
    convert_degrees_to_gps_coordinate,
    convert_gps_coordinate_to_degrees,
    read_exif_location,
)


class TestGPSUtilities:
    """Test GPS utility functions."""

    @pytest.fixture
    def test_image_path(self, tmp_path):
        """Create a test image path."""
        return tmp_path / "test_image.jpg"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "gps_coordinate,expected,description",
        [
            (
                [(37, 1), (46, 1), (30, 1)],  # Basic format: 37°46'30" = 37.775°
                37 + 46 / 60 + 30 / 3600,
                "basic integer coordinates",
            ),
            (
                [(374, 10), (2760, 100), (0, 1)],  # Fractions: 37.4° + 27.6' = 37.86°
                37.4 + 27.6 / 60 + 0 / 3600,
                "fractional coordinates like real EXIF data",
            ),
            (
                ((37, 1), (46, 1), (30, 1)),  # Tuple format: same as first
                37 + 46 / 60 + 30 / 3600,
                "tuple of tuples format",
            ),
        ],
    )
    def test_convert_gps_coordinate_to_degrees(self, gps_coordinate, expected, description):
        """Test converting GPS coordinates to degrees with various input formats."""
        result = convert_gps_coordinate_to_degrees(gps_coordinate)
        assert abs(result - expected) < 0.0001, f"Failed for {description}"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "degrees,expected_d,expected_m,expected_s_approx,description",
        [
            (37.775, 37, 46, 30000, "positive decimal with seconds"),  # 37°46'30"
            (-122.4194, 122, 25, None, "negative decimal (uses absolute value)"),  # Should use abs
            (0.0, 0, 0, 0, "zero degrees"),
            (1.5, 1, 30, 0, "precise half degree (30 minutes)"),
        ],
    )
    def test_convert_degrees_to_gps_coordinate(self, degrees, expected_d, expected_m, expected_s_approx, description):
        """Test converting decimal degrees to DMS format."""
        d, m, s = convert_degrees_to_gps_coordinate(degrees)

        assert d == expected_d, f"Degrees mismatch for {description}"
        assert m == expected_m, f"Minutes mismatch for {description}"

        if expected_s_approx is None:
            assert s > 0, f"Expected positive seconds for {description}"
        else:
            if expected_s_approx == 0:
                assert s == 0, f"Expected zero seconds for {description}"
            else:
                # Allow for small floating point precision differences
                assert abs(s - expected_s_approx) <= 1, f"Seconds precision issue for {description}"

    @pytest.mark.integration
    @patch("exiftool.ExifToolHelper")
    def test_read_exif_location_with_coordinates(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location from EXIF data."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        metadata = {
            "Composite:GPSLatitude": 37.7749,
            "Composite:GPSLongitude": -122.4194,
        }
        mock_et.get_metadata.return_value = [metadata]

        lat, lon = read_exif_location(test_image_path)

        assert lat == 37.7749
        assert lon == -122.4194
        mock_et.get_metadata.assert_called_once_with(str(test_image_path.absolute()))

    @pytest.mark.integration
    @patch("exiftool.ExifToolHelper")
    def test_read_exif_location_exif_fallback(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location using EXIF fallback when Composite tags missing."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        metadata = {
            "EXIF:GPSLatitude": 40.7128,
            "EXIF:GPSLongitude": -74.0060,
        }
        mock_et.get_metadata.return_value = [metadata]

        lat, lon = read_exif_location(test_image_path)

        assert lat == 40.7128
        assert lon == -74.0060

    @pytest.mark.integration
    @patch("exiftool.ExifToolHelper")
    def test_read_exif_location_partial_coordinates(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location with only partial coordinates."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        # Only latitude, no longitude
        metadata = {
            "Composite:GPSLatitude": 37.7749,
        }
        mock_et.get_metadata.return_value = [metadata]

        lat, lon = read_exif_location(test_image_path)

        assert lat is None
        assert lon is None

    @pytest.mark.integration
    @patch("exiftool.ExifToolHelper")
    def test_read_exif_location_no_metadata(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location when no metadata available."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        mock_et.get_metadata.return_value = []  # No metadata

        lat, lon = read_exif_location(test_image_path)

        assert lat is None
        assert lon is None

    @pytest.mark.integration
    @patch("exiftool.ExifToolHelper")
    def test_read_exif_location_empty_metadata(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location when metadata is None."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        mock_et.get_metadata.return_value = None  # None metadata

        lat, lon = read_exif_location(test_image_path)

        assert lat is None
        assert lon is None

    @pytest.mark.unit
    @patch("exiftool.ExifToolHelper")
    def test_read_exif_location_no_gps_data(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location when no GPS data in metadata."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        metadata = {
            "FileName": "test.jpg",
            "ImageWidth": 800,
            "ImageHeight": 600,
        }
        mock_et.get_metadata.return_value = [metadata]

        lat, lon = read_exif_location(test_image_path)

        assert lat is None
        assert lon is None

    @pytest.mark.integration
    @patch("exiftool.ExifToolHelper")
    @patch("marimba.lib.gps.show_dependency_error_and_exit")
    def test_read_exif_location_exiftool_not_found(self, mock_show_error, mock_exiftool_helper, test_image_path):
        """Test handling when exiftool is not found."""
        mock_exiftool_helper.side_effect = FileNotFoundError("exiftool not found")

        lat, lon = read_exif_location(test_image_path)

        assert lat is None
        assert lon is None
        mock_show_error.assert_called_once()

    @pytest.mark.integration
    @patch("exiftool.ExifToolHelper")
    def test_read_exif_location_file_not_found_other(self, mock_exiftool_helper, test_image_path):
        """Test handling FileNotFoundError not related to exiftool."""
        mock_exiftool_helper.side_effect = FileNotFoundError("Image file not found")

        lat, lon = read_exif_location(test_image_path)

        assert lat is None
        assert lon is None

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "exception,description",
        [
            (KeyError("Missing GPS data"), "missing GPS key"),
            (ValueError("Invalid coordinate format"), "invalid coordinate format"),
            (TypeError("Unexpected data type"), "unexpected data type"),
            (AttributeError("Missing attribute"), "missing attribute"),
            (IndexError("Index out of range"), "index out of range"),
        ],
    )
    @patch("exiftool.ExifToolHelper")
    def test_read_exif_location_various_exceptions(self, mock_exiftool_helper, test_image_path, exception, description):
        """Test handling various exceptions during GPS reading."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et
        mock_et.get_metadata.side_effect = exception

        lat, lon = read_exif_location(test_image_path)

        assert lat is None, f"Expected None latitude for {description}"
        assert lon is None, f"Expected None longitude for {description}"

    @pytest.mark.integration
    @patch("exiftool.ExifToolHelper")
    def test_read_exif_location_string_coordinates(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location when coordinates are strings."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        metadata = {
            "Composite:GPSLatitude": "37.7749",  # String values
            "Composite:GPSLongitude": "-122.4194",
        }
        mock_et.get_metadata.return_value = [metadata]

        lat, lon = read_exif_location(test_image_path)

        assert lat == 37.7749
        assert lon == -122.4194

    @pytest.mark.integration
    @patch("exiftool.ExifToolHelper")
    def test_read_exif_location_with_path_object(self, mock_exiftool_helper, tmp_path):
        """Test reading GPS location with Path object."""
        test_path = tmp_path / "subfolder" / "test.jpg"

        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et

        metadata = {
            "Composite:GPSLatitude": 51.5074,
            "Composite:GPSLongitude": -0.1278,
        }
        mock_et.get_metadata.return_value = [metadata]

        lat, lon = read_exif_location(test_path)

        assert lat == 51.5074
        assert lon == -0.1278
        # Should call with absolute path
        mock_et.get_metadata.assert_called_once_with(str(test_path.absolute()))
