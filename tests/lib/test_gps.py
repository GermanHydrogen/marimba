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

    def test_convert_gps_coordinate_to_degrees_basic(self):
        """Test converting GPS coordinates to degrees."""
        # 37 degrees, 46 minutes, 30 seconds = 37.775 degrees
        gps_coordinate = [
            (37, 1),    # degrees
            (46, 1),    # minutes  
            (30, 1),    # seconds
        ]
        
        result = convert_gps_coordinate_to_degrees(gps_coordinate)
        
        expected = 37 + 46/60 + 30/3600
        assert abs(result - expected) < 0.0001

    def test_convert_gps_coordinate_to_degrees_fractions(self):
        """Test converting GPS coordinates with fractional values."""
        # Using fractions like real EXIF data
        gps_coordinate = [
            (374, 10),     # 37.4 degrees
            (2760, 100),   # 27.6 minutes
            (0, 1),        # 0 seconds
        ]
        
        result = convert_gps_coordinate_to_degrees(gps_coordinate)
        
        expected = 37.4 + 27.6/60 + 0/3600
        assert abs(result - expected) < 0.0001

    def test_convert_gps_coordinate_to_degrees_tuple_format(self):
        """Test converting GPS coordinates in tuple format."""
        # Test with tuple of tuples format
        gps_coordinate = (
            (37, 1),    # degrees
            (46, 1),    # minutes  
            (30, 1),    # seconds
        )
        
        result = convert_gps_coordinate_to_degrees(gps_coordinate)
        
        expected = 37 + 46/60 + 30/3600
        assert abs(result - expected) < 0.0001

    def test_convert_degrees_to_gps_coordinate_positive(self):
        """Test converting positive decimal degrees to DMS."""
        degrees = 37.775  # 37 degrees, 46 minutes, 30 seconds
        
        d, m, s = convert_degrees_to_gps_coordinate(degrees)
        
        assert d == 37
        assert m == 46
        # Allow for small floating point precision differences
        assert abs(s - 30000) <= 1  # seconds * 1000, with tolerance

    def test_convert_degrees_to_gps_coordinate_negative(self):
        """Test converting negative decimal degrees to DMS."""
        degrees = -122.4194  # Should use absolute value
        
        d, m, s = convert_degrees_to_gps_coordinate(degrees)
        
        assert d == 122
        assert m == 25  # (0.4194 * 60) = 25.164, int() = 25
        assert s > 0     # Should have some seconds value

    def test_convert_degrees_to_gps_coordinate_zero(self):
        """Test converting zero degrees to DMS."""
        degrees = 0.0
        
        d, m, s = convert_degrees_to_gps_coordinate(degrees)
        
        assert d == 0
        assert m == 0
        assert s == 0

    def test_convert_degrees_to_gps_coordinate_precision(self):
        """Test precision of degrees to DMS conversion."""
        degrees = 1.5  # 1 degree, 30 minutes, 0 seconds
        
        d, m, s = convert_degrees_to_gps_coordinate(degrees)
        
        assert d == 1
        assert m == 30
        assert s == 0

    @patch('exiftool.ExifToolHelper')
    def test_read_exif_location_with_coordinates(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location from EXIF data."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et
        
        metadata = {
            'Composite:GPSLatitude': 37.7749,
            'Composite:GPSLongitude': -122.4194,
        }
        mock_et.get_metadata.return_value = [metadata]
        
        lat, lon = read_exif_location(test_image_path)
        
        assert lat == 37.7749
        assert lon == -122.4194
        mock_et.get_metadata.assert_called_once_with(str(test_image_path.absolute()))

    @patch('exiftool.ExifToolHelper')
    def test_read_exif_location_exif_fallback(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location using EXIF fallback when Composite tags missing."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et
        
        metadata = {
            'EXIF:GPSLatitude': 40.7128,
            'EXIF:GPSLongitude': -74.0060,
        }
        mock_et.get_metadata.return_value = [metadata]
        
        lat, lon = read_exif_location(test_image_path)
        
        assert lat == 40.7128
        assert lon == -74.0060

    @patch('exiftool.ExifToolHelper')
    def test_read_exif_location_partial_coordinates(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location with only partial coordinates."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et
        
        # Only latitude, no longitude
        metadata = {
            'Composite:GPSLatitude': 37.7749,
        }
        mock_et.get_metadata.return_value = [metadata]
        
        lat, lon = read_exif_location(test_image_path)
        
        assert lat is None
        assert lon is None

    @patch('exiftool.ExifToolHelper')
    def test_read_exif_location_no_metadata(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location when no metadata available."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et
        
        mock_et.get_metadata.return_value = []  # No metadata
        
        lat, lon = read_exif_location(test_image_path)
        
        assert lat is None
        assert lon is None

    @patch('exiftool.ExifToolHelper')
    def test_read_exif_location_empty_metadata(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location when metadata is None."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et
        
        mock_et.get_metadata.return_value = None  # None metadata
        
        lat, lon = read_exif_location(test_image_path)
        
        assert lat is None
        assert lon is None

    @patch('exiftool.ExifToolHelper')
    def test_read_exif_location_no_gps_data(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location when no GPS data in metadata."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et
        
        metadata = {
            'FileName': 'test.jpg',
            'ImageWidth': 800,
            'ImageHeight': 600,
        }
        mock_et.get_metadata.return_value = [metadata]
        
        lat, lon = read_exif_location(test_image_path)
        
        assert lat is None
        assert lon is None

    @patch('exiftool.ExifToolHelper')
    @patch('marimba.lib.gps.show_dependency_error_and_exit')
    def test_read_exif_location_exiftool_not_found(self, mock_show_error, mock_exiftool_helper, test_image_path):
        """Test handling when exiftool is not found."""
        mock_exiftool_helper.side_effect = FileNotFoundError("exiftool not found")
        
        lat, lon = read_exif_location(test_image_path)
        
        assert lat is None
        assert lon is None
        mock_show_error.assert_called_once()

    @patch('exiftool.ExifToolHelper')
    def test_read_exif_location_file_not_found_other(self, mock_exiftool_helper, test_image_path):
        """Test handling FileNotFoundError not related to exiftool."""
        mock_exiftool_helper.side_effect = FileNotFoundError("Image file not found")
        
        lat, lon = read_exif_location(test_image_path)
        
        assert lat is None
        assert lon is None

    @patch('exiftool.ExifToolHelper')
    def test_read_exif_location_various_exceptions(self, mock_exiftool_helper, test_image_path):
        """Test handling various exceptions during GPS reading."""
        exception_types = [
            KeyError("Missing GPS data"),
            ValueError("Invalid coordinate format"),
            TypeError("Unexpected data type"),
            AttributeError("Missing attribute"),
            IndexError("Index out of range"),
        ]
        
        for exception in exception_types:
            mock_et = Mock()
            mock_exiftool_helper.return_value.__enter__.return_value = mock_et
            mock_et.get_metadata.side_effect = exception
            
            lat, lon = read_exif_location(test_image_path)
            
            assert lat is None
            assert lon is None

    @patch('exiftool.ExifToolHelper')
    def test_read_exif_location_string_coordinates(self, mock_exiftool_helper, test_image_path):
        """Test reading GPS location when coordinates are strings."""
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et
        
        metadata = {
            'Composite:GPSLatitude': "37.7749",  # String values
            'Composite:GPSLongitude': "-122.4194",
        }
        mock_et.get_metadata.return_value = [metadata]
        
        lat, lon = read_exif_location(test_image_path)
        
        assert lat == 37.7749
        assert lon == -122.4194

    @patch('exiftool.ExifToolHelper')
    def test_read_exif_location_with_path_object(self, mock_exiftool_helper, tmp_path):
        """Test reading GPS location with Path object."""
        test_path = tmp_path / "subfolder" / "test.jpg"
        
        mock_et = Mock()
        mock_exiftool_helper.return_value.__enter__.return_value = mock_et
        
        metadata = {
            'Composite:GPSLatitude': 51.5074,
            'Composite:GPSLongitude': -0.1278,
        }
        mock_et.get_metadata.return_value = [metadata]
        
        lat, lon = read_exif_location(test_path)
        
        assert lat == 51.5074
        assert lon == -0.1278
        # Should call with absolute path
        mock_et.get_metadata.assert_called_once_with(str(test_path.absolute()))