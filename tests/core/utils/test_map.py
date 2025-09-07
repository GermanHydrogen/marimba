"""Tests for marimba.core.utils.map module."""

import pytest

from marimba.core.utils.map import (
    NetworkConnectionError,
    add_axes,
    calculate_grid_intervals,
    calculate_visible_bounds,
    calculate_zoom_level,
    lat_to_y,
    lon_to_x,
    make_summary_map,
    x_to_lon,
    y_to_lat,
)


class TestCoordinateConversions:
    """Test coordinate conversion functions."""

    @pytest.mark.unit
    def test_lat_to_y_basic(self):
        """Test latitude to y tile coordinate conversion."""
        # Test equator (0 degrees) at zoom level 1
        y = lat_to_y(0.0, 1)
        assert abs(y - 1.0) < 0.0001

        # Test north pole approximation
        y = lat_to_y(85.0, 1)
        assert y < 0.5

        # Test south pole approximation
        y = lat_to_y(-85.0, 1)
        assert y > 1.5

    @pytest.mark.unit
    def test_y_to_lat_basic(self):
        """Test y tile coordinate to latitude conversion."""
        # Test center tile at zoom level 1
        lat = y_to_lat(1.0, 1)
        assert abs(lat) < 0.0001

        # Test round-trip conversion
        original_lat = 45.0
        y = lat_to_y(original_lat, 10)
        converted_lat = y_to_lat(y, 10)
        assert abs(original_lat - converted_lat) < 0.0001

    @pytest.mark.unit
    def test_lon_to_x_basic(self):
        """Test longitude to x tile coordinate conversion."""
        # Test 0 degrees longitude at zoom level 1
        x = lon_to_x(0.0, 1)
        assert abs(x - 1.0) < 0.0001

        # Test 180 degrees longitude
        x = lon_to_x(180.0, 1)
        assert abs(x - 2.0) < 0.0001

        # Test -180 degrees longitude
        x = lon_to_x(-180.0, 1)
        assert abs(x) < 0.0001

    @pytest.mark.unit
    def test_x_to_lon_basic(self):
        """Test x tile coordinate to longitude conversion."""
        # Test center tile at zoom level 1
        lon = x_to_lon(1.0, 1)
        assert abs(lon) < 0.0001

        # Test round-trip conversion
        original_lon = -122.5
        x = lon_to_x(original_lon, 10)
        converted_lon = x_to_lon(x, 10)
        assert abs(original_lon - converted_lon) < 0.0001

    @pytest.mark.unit
    def test_coordinate_conversions_different_zoom_levels(self):
        """Test coordinate conversions at different zoom levels."""
        lat, lon = 37.7749, -122.4194  # San Francisco

        for zoom in [1, 5, 10, 15, 18]:
            # Test round-trip conversions
            x = lon_to_x(lon, zoom)
            y = lat_to_y(lat, zoom)

            converted_lon = x_to_lon(x, zoom)
            converted_lat = y_to_lat(y, zoom)

            assert abs(lat - converted_lat) < 0.001
            assert abs(lon - converted_lon) < 0.001

    @pytest.mark.unit
    def test_coordinate_edge_cases(self):
        """Test coordinate conversion edge cases."""
        # Test extreme latitudes
        extreme_lat = 89.99
        y = lat_to_y(extreme_lat, 1)
        converted_lat = y_to_lat(y, 1)
        assert abs(extreme_lat - converted_lat) < 0.1

        # Test extreme longitudes
        extreme_lon = 179.99
        x = lon_to_x(extreme_lon, 1)
        converted_lon = x_to_lon(x, 1)
        assert abs(extreme_lon - converted_lon) < 0.001


class TestGridCalculations:
    """Test grid interval calculation functions."""

    @pytest.mark.unit
    def test_calculate_grid_intervals_normal_range(self):
        """Test grid interval calculation for normal coordinate range."""
        positions, decimals = calculate_grid_intervals(-1.0, 1.0, 3)

        assert len(positions) == 5  # 3 + 2
        assert positions[0] == -1.0
        assert positions[-1] == 1.0
        assert decimals >= 2

    @pytest.mark.unit
    def test_calculate_grid_intervals_small_range(self):
        """Test grid interval calculation for very small coordinate range."""
        # Very small range should be expanded
        positions, decimals = calculate_grid_intervals(0.0, 1e-12, 3)

        assert len(positions) == 5
        assert decimals >= 2
        # Range should be expanded to DEFAULT_SMALL_RANGE
        total_range = positions[-1] - positions[0]
        assert total_range > 1e-12

    @pytest.mark.unit
    def test_calculate_grid_intervals_decimal_places(self):
        """Test decimal places calculation for different intervals."""
        # Large interval -> fewer decimal places
        positions, decimals = calculate_grid_intervals(0.0, 10.0, 3)
        assert decimals == 2

        # Medium interval (need larger range to get smaller interval)
        positions, decimals = calculate_grid_intervals(0.0, 0.02, 3)
        assert decimals == 3

        # Small interval
        positions, decimals = calculate_grid_intervals(0.0, 0.002, 3)
        assert decimals == 4

        # Tiny interval
        positions, decimals = calculate_grid_intervals(0.0, 0.0002, 3)
        assert decimals == 5

    @pytest.mark.unit
    def test_calculate_grid_intervals_different_num_lines(self):
        """Test grid calculation with different number of lines."""
        for num_lines in [1, 3, 5, 10]:
            positions, decimals = calculate_grid_intervals(0.0, 10.0, num_lines)
            assert len(positions) == num_lines + 2
            assert positions[0] == 0.0
            assert positions[-1] == 10.0

    @pytest.mark.unit
    def test_calculate_grid_intervals_negative_range(self):
        """Test grid calculation with negative coordinate range."""
        positions, decimals = calculate_grid_intervals(-5.0, -2.0, 2)

        assert len(positions) == 4
        assert positions[0] == -5.0
        assert positions[-1] == -2.0
        assert all(p1 < p2 for p1, p2 in zip(positions[:-1], positions[1:], strict=False))


class TestVisibleBounds:
    """Test visible bounds calculation."""

    @pytest.mark.unit
    def test_calculate_visible_bounds_basic(self):
        """Test basic visible bounds calculation."""
        center_lat, center_lon = 0.0, 0.0
        zoom = 1
        width, height = 512, 512

        min_lat, max_lat, min_lon, max_lon = calculate_visible_bounds(center_lat, center_lon, zoom, width, height)

        # At zoom level 1 with 512x512, should cover significant area
        assert min_lat < center_lat < max_lat
        assert min_lon < center_lon < max_lon
        assert max_lat - min_lat > 0
        assert max_lon - min_lon > 0

    @pytest.mark.unit
    def test_calculate_visible_bounds_different_sizes(self):
        """Test visible bounds with different map sizes."""
        center_lat, center_lon = 37.7749, -122.4194
        zoom = 10

        # Larger map should have larger bounds
        bounds_small = calculate_visible_bounds(center_lat, center_lon, zoom, 256, 256)
        bounds_large = calculate_visible_bounds(center_lat, center_lon, zoom, 1024, 1024)

        # Large map should cover more area
        small_lat_range = bounds_small[1] - bounds_small[0]
        large_lat_range = bounds_large[1] - bounds_large[0]
        assert large_lat_range > small_lat_range

    @pytest.mark.unit
    def test_calculate_visible_bounds_different_zoom(self):
        """Test visible bounds with different zoom levels."""
        center_lat, center_lon = 37.7749, -122.4194
        width, height = 512, 512

        # Lower zoom should have larger bounds
        bounds_low = calculate_visible_bounds(center_lat, center_lon, 5, width, height)
        bounds_high = calculate_visible_bounds(center_lat, center_lon, 15, width, height)

        low_lat_range = bounds_low[1] - bounds_low[0]
        high_lat_range = bounds_high[1] - bounds_high[0]
        assert low_lat_range > high_lat_range

    @pytest.mark.unit
    def test_calculate_visible_bounds_rectangular(self):
        """Test visible bounds with rectangular (non-square) dimensions."""
        center_lat, center_lon = 0.0, 0.0
        zoom = 10

        # Wide rectangle
        bounds_wide = calculate_visible_bounds(center_lat, center_lon, zoom, 1024, 256)
        # Tall rectangle
        bounds_tall = calculate_visible_bounds(center_lat, center_lon, zoom, 256, 1024)

        wide_lon_range = bounds_wide[3] - bounds_wide[2]
        wide_lat_range = bounds_wide[1] - bounds_wide[0]
        tall_lon_range = bounds_tall[3] - bounds_tall[2]
        tall_lat_range = bounds_tall[1] - bounds_tall[0]

        # Wide map should have larger longitude range
        assert wide_lon_range > tall_lon_range
        # Tall map should have larger latitude range
        assert tall_lat_range > wide_lat_range


class TestAxesDrawing:
    """Test axes drawing functionality."""

    @pytest.mark.unit
    def test_add_axes_basic(self, mocker):
        """Test basic axes drawing functionality."""
        mock_draw_instance = mocker.Mock()

        add_axes(
            mock_draw_instance,
            width=500,
            height=500,
            num_x_lines=3,
            num_y_lines=3,
            min_lat=37.0,
            max_lat=38.0,
            min_lon=-123.0,
            max_lon=-122.0,
            zoom=10,
        )

        # Should call draw methods (line and text)
        assert mock_draw_instance.line.called
        assert mock_draw_instance.text.called

    @pytest.mark.unit
    def test_add_axes_no_lines(self, mocker):
        """Test axes drawing with no grid lines."""
        mock_draw_instance = mocker.Mock()

        add_axes(
            mock_draw_instance,
            width=500,
            height=500,
            num_x_lines=0,
            num_y_lines=0,
            min_lat=37.0,
            max_lat=38.0,
            min_lon=-123.0,
            max_lon=-122.0,
            zoom=10,
        )

        # Should still work but with minimal drawing
        # Grid interval calculation should still return valid results
        assert True  # Function should not raise an error

    @pytest.mark.unit
    def test_add_axes_large_decimal_precision(self, mocker):
        """Test axes drawing with high precision coordinates."""
        mock_draw_instance = mocker.Mock()

        # Very small coordinate range requiring high precision
        add_axes(
            mock_draw_instance,
            width=500,
            height=500,
            num_x_lines=3,
            num_y_lines=3,
            min_lat=37.7740,
            max_lat=37.7750,  # Very small range
            min_lon=-122.4200,
            max_lon=-122.4190,  # Very small range
            zoom=18,  # High zoom for precision
        )

        # Should handle high precision coordinates
        assert mock_draw_instance.line.called
        assert mock_draw_instance.text.called


class TestZoomCalculation:
    """Test zoom level calculation."""

    @pytest.mark.unit
    def test_calculate_zoom_level_basic(self):
        """Test basic zoom level calculation."""
        # Very large area should use low zoom
        zoom = calculate_zoom_level(0.0, 90.0, -180.0, 180.0, width=512, height=512)
        assert isinstance(zoom, int)
        assert 0 <= zoom <= 19

    @pytest.mark.unit
    def test_calculate_zoom_level_small_area(self):
        """Test zoom calculation for small area."""
        # Small area should use higher zoom
        zoom_small = calculate_zoom_level(37.77, 37.78, -122.42, -122.41, width=512, height=512)
        zoom_large = calculate_zoom_level(30.0, 40.0, -130.0, -120.0, width=512, height=512)

        assert zoom_small >= zoom_large

    @pytest.mark.unit
    def test_calculate_zoom_level_extreme_values(self):
        """Test zoom calculation with extreme coordinate values."""
        # Very small area should get high zoom (clamped to max)
        zoom_tiny = calculate_zoom_level(0.0, 0.00001, 0.0, 0.00001, width=512, height=512)
        assert 15 <= zoom_tiny <= 19

        # Very large area should get low zoom
        zoom_huge = calculate_zoom_level(-85.0, 85.0, -180.0, 180.0, width=512, height=512)
        assert 0 <= zoom_huge <= 5

    @pytest.mark.unit
    def test_calculate_zoom_level_different_dimensions(self):
        """Test zoom calculation with different image dimensions."""
        coords = (37.0, 38.0, -123.0, -122.0)

        # Larger image can accommodate higher zoom
        zoom_small = calculate_zoom_level(*coords, width=256, height=256)
        zoom_large = calculate_zoom_level(*coords, width=2048, height=2048)

        assert zoom_large >= zoom_small


class TestMapGeneration:
    """Test map generation functionality."""

    @pytest.mark.integration
    def test_make_summary_map_basic(self, mocker):
        """Test basic map generation."""
        # Mock successful network response
        mock_head = mocker.patch("marimba.core.utils.map.requests.head")
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        # Mock StaticMap
        mock_static_map = mocker.patch("marimba.core.utils.map.StaticMap")
        mock_map = mocker.Mock()
        mock_map.render.return_value = mocker.Mock()
        mock_static_map.return_value = mock_map

        # Mock PIL Image
        mock_image_new = mocker.patch("marimba.core.utils.map.Image.new")
        mock_img = mocker.Mock()
        mock_image_new.return_value = mock_img
        mock_draw = mocker.patch("marimba.core.utils.map.ImageDraw.Draw")
        mock_draw.return_value = mocker.Mock()

        coords = [(37.7749, -122.4194), (37.7849, -122.4094)]

        result = make_summary_map(coords, width=500, height=500)

        assert result is not None
        mock_static_map.assert_called_once()
        mock_map.render.assert_called_once()

    @pytest.mark.integration
    def test_make_summary_map_network_error(self, mocker):
        """Test map generation with network error."""
        # Mock StaticMap to raise ConnectionError
        mock_static_map = mocker.patch("marimba.core.utils.map.StaticMap")
        mock_map = mocker.Mock()
        mock_map.render.side_effect = __import__("requests").exceptions.ConnectionError("Network error")
        mock_static_map.return_value = mock_map

        coords = [(37.7749, -122.4194), (37.7849, -122.4094)]

        with pytest.raises(NetworkConnectionError):
            make_summary_map(coords, width=500, height=500)

    @pytest.mark.unit
    def test_make_summary_map_empty_coords(self, mocker):
        """Test map generation with empty coordinates."""
        mock_head = mocker.patch("marimba.core.utils.map.requests.head")
        mock_static_map = mocker.patch("marimba.core.utils.map.StaticMap")
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        # Should handle empty coordinates gracefully
        _ = make_summary_map([], width=500, height=500)

        # Should not create a map with no coordinates
        mock_static_map.assert_not_called()

    @pytest.mark.unit
    def test_make_summary_map_single_coordinate(self, mocker):
        """Test map generation with single coordinate."""
        mock_head = mocker.patch("marimba.core.utils.map.requests.head")
        mock_static_map = mocker.patch("marimba.core.utils.map.StaticMap")
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        mock_map = mocker.Mock()
        mock_map.render.return_value = mocker.Mock()
        mock_static_map.return_value = mock_map

        coords = [(37.7749, -122.4194)]  # Single coordinate

        _ = make_summary_map(coords, width=500, height=500)

        # Should handle single coordinate
        mock_static_map.assert_called_once()


class TestNetworkConnectionError:
    """Test custom exception class."""

    @pytest.mark.unit
    def test_network_connection_error_creation(self):
        """Test NetworkConnectionError can be created and raised."""
        with pytest.raises(NetworkConnectionError):
            raise NetworkConnectionError("Test error message")

    @pytest.mark.unit
    def test_network_connection_error_inheritance(self):
        """Test NetworkConnectionError inherits from Exception."""
        error = NetworkConnectionError("Test")
        assert isinstance(error, Exception)

    @pytest.mark.unit
    def test_network_connection_error_message(self):
        """Test NetworkConnectionError preserves message."""
        message = "Custom error message"
        error = NetworkConnectionError(message)
        assert str(error) == message
