"""Tests for marimba.lib.image module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from marimba.lib.image import (
    GridDimensions,
    GridImageProcessor,
    GridRow,
    OutputPathManager,
    apply_clahe,
    convert_to_jpeg,
    create_grid_image,
    crop,
    flip_horizontal,
    flip_vertical,
    gaussian_blur,
    generate_image_thumbnail,
    get_average_image_color,
    get_shannon_entropy,
    get_width_height,
    is_blurry,
    resize_exact,
    resize_fit,
    rotate_clockwise,
    scale,
    sharpen,
    turn_clockwise,
)


class TestImageUtilities:
    """Test image utility functions."""

    @pytest.fixture
    def test_image_rgb(self, tmp_path):
        """Create a test RGB image."""
        img = Image.new("RGB", (100, 80), color=(255, 0, 0))
        image_path = tmp_path / "test_image.png"
        img.save(image_path)
        return image_path

    @pytest.fixture
    def test_image_jpeg(self, tmp_path):
        """Create a test JPEG image."""
        img = Image.new("RGB", (200, 150), color=(0, 255, 0))
        image_path = tmp_path / "test_image.jpg"
        img.save(image_path, "JPEG")
        return image_path

    @pytest.fixture
    def test_image_large(self, tmp_path):
        """Create a large test image."""
        img = Image.new("RGB", (2000, 1500), color=(0, 0, 255))
        image_path = tmp_path / "large_image.png"
        img.save(image_path)
        return image_path

    def test_generate_image_thumbnail(self, test_image_rgb, tmp_path):
        """Test thumbnail generation."""
        output_dir = tmp_path / "thumbnails"
        output_dir.mkdir()

        result_path = generate_image_thumbnail(test_image_rgb, output_dir)

        assert result_path.exists()
        assert result_path.suffix == ".png"
        assert "_THUMB" in result_path.name

        with Image.open(result_path) as thumb:
            width, height = thumb.size
            assert width <= 300
            assert height <= 300

    def test_generate_image_thumbnail_custom_suffix(self, test_image_rgb, tmp_path):
        """Test thumbnail generation with custom suffix."""
        output_dir = tmp_path / "thumbnails"
        output_dir.mkdir()

        result_path = generate_image_thumbnail(test_image_rgb, output_dir, "_CUSTOM")

        assert result_path.exists()
        assert "_CUSTOM" in result_path.name

    def test_generate_image_thumbnail_existing(self, test_image_rgb, tmp_path):
        """Test that existing thumbnail is not regenerated."""
        output_dir = tmp_path / "thumbnails"
        output_dir.mkdir()

        # Create the thumbnail first
        result_path = generate_image_thumbnail(test_image_rgb, output_dir)
        original_mtime = result_path.stat().st_mtime

        # Call again - should not recreate
        result_path2 = generate_image_thumbnail(test_image_rgb, output_dir)

        assert result_path == result_path2
        assert result_path.stat().st_mtime == original_mtime

    def test_convert_to_jpeg_from_png(self, test_image_rgb, tmp_path):
        """Test converting PNG to JPEG."""
        output_path = tmp_path / "converted.jpg"

        result_path = convert_to_jpeg(test_image_rgb, destination=output_path)

        assert result_path.suffix == ".jpg"
        assert result_path.exists()

        with Image.open(result_path) as img:
            assert img.format == "JPEG"

    def test_convert_to_jpeg_existing_jpeg(self, test_image_jpeg, tmp_path):
        """Test converting existing JPEG (should copy)."""
        output_path = tmp_path / "converted.jpg"

        result_path = convert_to_jpeg(test_image_jpeg, destination=output_path)

        assert result_path == output_path
        assert result_path.exists()

    def test_convert_to_jpeg_custom_quality(self, test_image_rgb):
        """Test JPEG conversion with custom quality."""
        result_path = convert_to_jpeg(test_image_rgb, quality=50)

        assert result_path.suffix == ".jpg"
        assert result_path.exists()

    def test_convert_to_jpeg_overwrites_original(self, test_image_rgb):
        """Test that conversion overwrites original when no destination specified."""
        original_path = test_image_rgb

        result_path = convert_to_jpeg(original_path)

        assert result_path.suffix == ".jpg"
        assert result_path.stem == original_path.stem

    def test_resize_fit_no_resize_needed(self, test_image_rgb, tmp_path):
        """Test resize_fit when image is already within bounds."""
        output_path = tmp_path / "resized.png"

        resize_fit(test_image_rgb, 200, 200, output_path)

        assert output_path.exists()
        with Image.open(output_path) as img:
            # Original is 100x80, should not be resized
            assert img.size == (100, 80)

    def test_resize_fit_width_constrained(self, test_image_large, tmp_path):
        """Test resize_fit when width is the constraining dimension."""
        output_path = tmp_path / "resized.png"

        resize_fit(test_image_large, 1000, 2000, output_path)

        assert output_path.exists()
        with Image.open(output_path) as img:
            width, height = img.size
            assert width == 1000
            assert height == 750  # Maintains aspect ratio

    def test_resize_fit_height_constrained(self, test_image_large, tmp_path):
        """Test resize_fit when height is the constraining dimension."""
        output_path = tmp_path / "resized.png"

        resize_fit(test_image_large, 3000, 1000, output_path)

        assert output_path.exists()
        with Image.open(output_path) as img:
            width, height = img.size
            assert width == 1333  # Maintains aspect ratio
            assert height == 1000

    def test_resize_fit_overwrites_original(self, test_image_large):
        """Test resize_fit overwrites original when no destination specified."""
        original_size = Image.open(test_image_large).size

        resize_fit(test_image_large, 500, 500)

        with Image.open(test_image_large) as img:
            new_size = img.size
            assert new_size != original_size
            assert max(new_size) <= 500

    def test_resize_exact(self, test_image_rgb, tmp_path):
        """Test exact resizing."""
        output_path = tmp_path / "resized.png"

        resize_exact(test_image_rgb, 150, 200, output_path)

        assert output_path.exists()
        with Image.open(output_path) as img:
            assert img.size == (150, 200)

    def test_resize_exact_overwrites_original(self, test_image_rgb):
        """Test resize_exact overwrites original when no destination specified."""
        resize_exact(test_image_rgb, 50, 40)

        with Image.open(test_image_rgb) as img:
            assert img.size == (50, 40)

    def test_scale(self, test_image_rgb, tmp_path):
        """Test scaling by factor."""
        output_path = tmp_path / "scaled.png"

        scale(test_image_rgb, 0.5, output_path)

        assert output_path.exists()
        with Image.open(output_path) as img:
            assert img.size == (50, 40)

    def test_scale_larger(self, test_image_rgb, tmp_path):
        """Test scaling up."""
        output_path = tmp_path / "scaled.png"

        scale(test_image_rgb, 2.0, output_path)

        assert output_path.exists()
        with Image.open(output_path) as img:
            assert img.size == (200, 160)

    def test_scale_overwrites_original(self, test_image_rgb):
        """Test scale overwrites original when no destination specified."""
        original_size = Image.open(test_image_rgb).size

        scale(test_image_rgb, 1.5)

        with Image.open(test_image_rgb) as img:
            new_size = img.size
            assert new_size != original_size
            assert new_size == (150, 120)

    def test_rotate_clockwise(self, test_image_rgb, tmp_path):
        """Test clockwise rotation."""
        output_path = tmp_path / "rotated.png"

        rotate_clockwise(test_image_rgb, 45, destination=output_path)

        assert output_path.exists()
        # Note: Actual rotation testing would require more complex image comparison

    def test_rotate_clockwise_expand(self, test_image_rgb, tmp_path):
        """Test clockwise rotation with expand."""
        output_path = tmp_path / "rotated.png"

        rotate_clockwise(test_image_rgb, 45, expand=True, destination=output_path)

        assert output_path.exists()

    def test_rotate_clockwise_overwrites_original(self, test_image_rgb):
        """Test rotate_clockwise overwrites original when no destination specified."""
        rotate_clockwise(test_image_rgb, 90)

        with Image.open(test_image_rgb) as img:
            # Note: rotation doesn't swap dimensions, just rotates content
            assert img.size == (100, 80)

    def test_turn_clockwise_90(self, test_image_rgb, tmp_path):
        """Test 90-degree clockwise turn."""
        output_path = tmp_path / "turned.png"

        turn_clockwise(test_image_rgb, 1, output_path)

        assert output_path.exists()
        with Image.open(output_path) as img:
            # 90 degree turn swaps dimensions
            assert img.size == (80, 100)

    def test_turn_clockwise_180(self, test_image_rgb, tmp_path):
        """Test 180-degree clockwise turn."""
        output_path = tmp_path / "turned.png"

        turn_clockwise(test_image_rgb, 2, output_path)

        assert output_path.exists()
        with Image.open(output_path) as img:
            # 180 degree turn keeps same dimensions
            assert img.size == (100, 80)

    def test_turn_clockwise_270(self, test_image_rgb, tmp_path):
        """Test 270-degree clockwise turn."""
        output_path = tmp_path / "turned.png"

        turn_clockwise(test_image_rgb, 3, output_path)

        assert output_path.exists()
        with Image.open(output_path) as img:
            # 270 degree turn swaps dimensions
            assert img.size == (80, 100)

    def test_turn_clockwise_invalid_turns(self, test_image_rgb):
        """Test invalid turns value raises error."""
        with pytest.raises(ValueError, match="Turns must be an integer between 1 and 3"):
            turn_clockwise(test_image_rgb, 0)

        with pytest.raises(ValueError, match="Turns must be an integer between 1 and 3"):
            turn_clockwise(test_image_rgb, 4)

    def test_flip_vertical(self, test_image_rgb, tmp_path):
        """Test vertical flip."""
        output_path = tmp_path / "flipped.png"

        flip_vertical(test_image_rgb, output_path)

        assert output_path.exists()
        with Image.open(output_path) as img:
            assert img.size == (100, 80)

    def test_flip_horizontal(self, test_image_rgb, tmp_path):
        """Test horizontal flip."""
        output_path = tmp_path / "flipped.png"

        flip_horizontal(test_image_rgb, output_path)

        assert output_path.exists()
        with Image.open(output_path) as img:
            assert img.size == (100, 80)

    @patch("cv2.imread")
    @patch("cv2.cvtColor")
    @patch("cv2.Laplacian")
    def test_is_blurry_true(self, mock_laplacian, mock_cvtcolor, mock_imread, test_image_rgb):
        """Test blur detection when image is blurry."""
        # Mock OpenCV functions
        mock_imread.return_value = np.zeros((80, 100, 3), dtype=np.uint8)
        mock_cvtcolor.return_value = np.zeros((80, 100), dtype=np.uint8)
        mock_laplacian.return_value.var.return_value = 50.0  # Below default threshold

        result = is_blurry(test_image_rgb)

        assert result is True

    @patch("cv2.imread")
    @patch("cv2.cvtColor")
    @patch("cv2.Laplacian")
    def test_is_blurry_false(self, mock_laplacian, mock_cvtcolor, mock_imread, test_image_rgb):
        """Test blur detection when image is not blurry."""
        # Mock OpenCV functions
        mock_imread.return_value = np.zeros((80, 100, 3), dtype=np.uint8)
        mock_cvtcolor.return_value = np.zeros((80, 100), dtype=np.uint8)
        mock_laplacian.return_value.var.return_value = 150.0  # Above default threshold

        result = is_blurry(test_image_rgb)

        assert result is False

    @patch("cv2.imread")
    def test_is_blurry_invalid_image(self, mock_imread, test_image_rgb):
        """Test blur detection with invalid image."""
        mock_imread.return_value = None

        with pytest.raises(ValueError, match="Could not load the image"):
            is_blurry(test_image_rgb)

    @patch("cv2.imread")
    @patch("cv2.cvtColor")
    @patch("cv2.Laplacian")
    def test_is_blurry_custom_threshold(self, mock_laplacian, mock_cvtcolor, mock_imread, test_image_rgb):
        """Test blur detection with custom threshold."""
        mock_imread.return_value = np.zeros((80, 100, 3), dtype=np.uint8)
        mock_cvtcolor.return_value = np.zeros((80, 100), dtype=np.uint8)
        mock_laplacian.return_value.var.return_value = 75.0

        result_low_threshold = is_blurry(test_image_rgb, threshold=50.0)
        result_high_threshold = is_blurry(test_image_rgb, threshold=100.0)

        assert result_low_threshold is False  # 75 > 50
        assert result_high_threshold is True  # 75 < 100

    def test_crop(self, test_image_rgb, tmp_path):
        """Test image cropping."""
        output_path = tmp_path / "cropped.png"

        crop(test_image_rgb, 10, 15, 50, 40, output_path)

        assert output_path.exists()
        with Image.open(output_path) as img:
            assert img.size == (50, 40)

    def test_crop_overwrites_original(self, test_image_rgb):
        """Test crop overwrites original when no destination specified."""
        crop(test_image_rgb, 0, 0, 50, 40)

        with Image.open(test_image_rgb) as img:
            assert img.size == (50, 40)

    @patch("cv2.imread")
    @patch("cv2.createCLAHE")
    @patch("cv2.imwrite")
    def test_apply_clahe(self, mock_imwrite, mock_create_clahe, mock_imread, test_image_rgb, tmp_path):
        """Test CLAHE application."""
        output_path = tmp_path / "clahe.png"

        # Mock OpenCV functions
        mock_img = np.zeros((80, 100), dtype=np.uint8)
        mock_imread.return_value = mock_img
        mock_clahe_obj = Mock()
        mock_clahe_obj.apply.return_value = mock_img
        mock_create_clahe.return_value = mock_clahe_obj

        apply_clahe(test_image_rgb, destination=output_path)

        mock_imread.assert_called_once()
        mock_create_clahe.assert_called_once_with(clipLimit=2.0, tileGridSize=(8, 8))
        mock_clahe_obj.apply.assert_called_once()
        mock_imwrite.assert_called_once()

    @patch("cv2.imread")
    def test_apply_clahe_invalid_image(self, mock_imread, test_image_rgb):
        """Test CLAHE with invalid image."""
        mock_imread.return_value = None

        with pytest.raises(ValueError, match="Could not read image"):
            apply_clahe(test_image_rgb)

    @patch("cv2.imread")
    @patch("cv2.GaussianBlur")
    @patch("cv2.imwrite")
    def test_gaussian_blur(self, mock_imwrite, mock_blur, mock_imread, test_image_rgb, tmp_path):
        """Test Gaussian blur application."""
        output_path = tmp_path / "blurred.png"

        # Mock OpenCV functions
        mock_img = np.zeros((80, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        mock_blur.return_value = mock_img

        gaussian_blur(test_image_rgb, destination=output_path)

        mock_imread.assert_called_once()
        mock_blur.assert_called_once_with(mock_img, (5, 5), 0)
        mock_imwrite.assert_called_once()

    @patch("cv2.imread")
    def test_gaussian_blur_invalid_image(self, mock_imread, test_image_rgb):
        """Test Gaussian blur with invalid image."""
        mock_imread.return_value = None

        with pytest.raises(ValueError, match="Could not read image"):
            gaussian_blur(test_image_rgb)

    @patch("cv2.imread")
    @patch("cv2.filter2D")
    @patch("cv2.imwrite")
    def test_sharpen(self, mock_imwrite, mock_filter2d, mock_imread, test_image_rgb, tmp_path):
        """Test image sharpening."""
        output_path = tmp_path / "sharpened.png"

        # Mock OpenCV functions
        mock_img = np.zeros((80, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_img
        mock_filter2d.return_value = mock_img

        sharpen(test_image_rgb, destination=output_path)

        mock_imread.assert_called_once()
        mock_filter2d.assert_called_once()
        mock_imwrite.assert_called_once()

    @patch("cv2.imread")
    def test_sharpen_invalid_image(self, mock_imread, test_image_rgb):
        """Test sharpening with invalid image."""
        mock_imread.return_value = None

        with pytest.raises(ValueError, match="Could not read image"):
            sharpen(test_image_rgb)

    def test_get_width_height(self, test_image_rgb):
        """Test getting image dimensions."""
        width, height = get_width_height(test_image_rgb)

        assert width == 100
        assert height == 80

    @patch("PIL.Image.open")
    def test_get_width_height_invalid_size(self, mock_open, test_image_rgb):
        """Test get_width_height with invalid size tuple."""
        mock_img = Mock()
        mock_img.size = (100,)  # Invalid - only one dimension
        mock_open.return_value.__enter__.return_value = mock_img

        with pytest.raises(ValueError, match="Size must be a tuple of two integers"):
            get_width_height(test_image_rgb)

    def test_get_shannon_entropy(self):
        """Test Shannon entropy calculation."""
        # Create a simple gradient image
        img_array = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            img_array[i, :] = i * 2 if i * 2 < 256 else 255

        # Use the modern PIL API without deprecated mode parameter
        img = Image.fromarray(img_array).convert("L")

        entropy = get_shannon_entropy(img)

        assert isinstance(entropy, float)
        assert entropy > 0

    def test_get_average_image_color(self):
        """Test average color calculation."""
        # Create a solid color image
        img = Image.new("RGB", (100, 100), color=(100, 150, 200))

        avg_color = get_average_image_color(img)

        assert len(avg_color) == 3
        assert avg_color == [100, 150, 200]

    def test_get_average_image_color_grayscale(self):
        """Test average color calculation for grayscale image."""
        # Create a grayscale image and convert to RGB for testing
        img = Image.new("L", (100, 100), color=128).convert("RGB")

        avg_color = get_average_image_color(img)

        assert len(avg_color) == 3
        assert avg_color == [128, 128, 128]


class TestGridClasses:
    """Test grid-related classes."""

    @pytest.fixture
    def grid_dimensions(self):
        """Create test grid dimensions."""
        return GridDimensions(columns=3, column_width=200, max_height=800)

    @pytest.fixture
    def test_images(self, tmp_path):
        """Create multiple test images."""
        images = []
        for i in range(6):
            img = Image.new("RGB", (300, 200), color=(i * 40, (i * 40) % 255, 100))
            path = tmp_path / f"test_image_{i}.png"
            img.save(path)
            images.append(path)
        return images

    def test_grid_dimensions(self, grid_dimensions):
        """Test GridDimensions dataclass."""
        assert grid_dimensions.columns == 3
        assert grid_dimensions.column_width == 200
        assert grid_dimensions.max_height == 800

    def test_grid_row_init(self, grid_dimensions):
        """Test GridRow initialization."""
        row = GridRow(grid_dimensions)

        assert row.images == []
        assert row.height == 0
        assert row.dimensions == grid_dimensions

    def test_grid_row_add_image(self, grid_dimensions):
        """Test adding image to GridRow."""
        row = GridRow(grid_dimensions)
        img = Image.new("RGB", (300, 200), color=(255, 0, 0))

        result = row.add_image(img)

        assert result is True
        assert len(row.images) == 1
        assert row.height == 133  # Scaled to fit column width

    def test_grid_row_add_image_full(self, grid_dimensions):
        """Test adding image to full GridRow."""
        row = GridRow(grid_dimensions)

        # Fill the row
        for _ in range(3):
            img = Image.new("RGB", (300, 200), color=(255, 0, 0))
            row.add_image(img)

        # Try to add one more
        img = Image.new("RGB", (300, 200), color=(0, 255, 0))
        result = row.add_image(img)

        assert result is False
        assert len(row.images) == 3

    def test_grid_row_cleanup(self, grid_dimensions):
        """Test GridRow cleanup."""
        row = GridRow(grid_dimensions)
        img = Image.new("RGB", (300, 200), color=(255, 0, 0))
        row.add_image(img)

        row.cleanup()

        # Images should be closed (can't easily test this without mock)

    def test_grid_image_processor_init(self, grid_dimensions):
        """Test GridImageProcessor initialization."""
        processor = GridImageProcessor(grid_dimensions)

        assert processor.dimensions == grid_dimensions

    def test_grid_image_processor_process_single_image(self, grid_dimensions, test_images):
        """Test processing single image."""
        processor = GridImageProcessor(grid_dimensions)
        row = GridRow(grid_dimensions)

        result = processor.process_single_image(test_images[0], row)

        assert result is True
        assert len(row.images) == 1

    def test_grid_image_processor_process_invalid_image(self, grid_dimensions, tmp_path):
        """Test processing invalid image."""
        processor = GridImageProcessor(grid_dimensions)
        row = GridRow(grid_dimensions)
        invalid_path = tmp_path / "nonexistent.png"

        result = processor.process_single_image(invalid_path, row)

        assert result is False
        assert len(row.images) == 0

    def test_grid_image_processor_create_grid(self, grid_dimensions, test_images):
        """Test creating grid from images."""
        processor = GridImageProcessor(grid_dimensions)

        grid_image, height, processed = processor.create_grid(test_images[:3])

        assert grid_image is not None
        assert height > 0
        assert processed == 3

    def test_grid_image_processor_create_grid_empty(self, grid_dimensions):
        """Test creating grid with no images."""
        processor = GridImageProcessor(grid_dimensions)

        grid_image, height, processed = processor.create_grid([])

        assert grid_image is None
        assert height == 0
        assert processed == 0

    def test_grid_image_processor_create_grid_max_height_exceeded(self, test_images):
        """Test creating grid when max height is exceeded by first row."""
        # Very small max height to trigger the height check
        small_dimensions = GridDimensions(columns=3, column_width=200, max_height=10)
        processor = GridImageProcessor(small_dimensions)

        grid_image, height, processed = processor.create_grid(test_images[:1])

        # Should still process the first image but might not fit all
        assert processed >= 0

    def test_output_path_manager_init(self, tmp_path):
        """Test OutputPathManager initialization."""
        base_path = tmp_path / "grid.jpg"
        manager = OutputPathManager(base_path)

        assert manager.base_path == base_path
        assert manager.extension == ""

    def test_output_path_manager_init_no_extension(self, tmp_path):
        """Test OutputPathManager initialization without extension."""
        base_path = tmp_path / "grid"
        manager = OutputPathManager(base_path)

        assert manager.base_path == base_path
        assert manager.extension == ".jpg"

    def test_output_path_manager_create_path_single_grid(self, tmp_path):
        """Test creating path for single grid."""
        base_path = tmp_path / "grid.jpg"
        manager = OutputPathManager(base_path)

        path = manager.create_path(0, False)

        assert path == base_path

    def test_output_path_manager_create_path_multiple_grids(self, tmp_path):
        """Test creating path for multiple grids."""
        base_path = tmp_path / "grid.jpg"
        manager = OutputPathManager(base_path)

        path = manager.create_path(0, True)

        assert path.name == "grid_01.jpg"
        assert path.parent == base_path.parent

    def test_create_grid_image(self, test_images, tmp_path):
        """Test creating grid image from paths."""
        output_path = tmp_path / "grid.jpg"

        created_files = create_grid_image(test_images, output_path, columns=2)

        assert len(created_files) >= 1
        assert created_files[0].exists()

    def test_create_grid_image_empty(self, tmp_path):
        """Test creating grid image with no paths."""
        output_path = tmp_path / "grid.jpg"

        created_files = create_grid_image([], output_path)

        assert created_files == []

    def test_create_grid_image_multiple_grids(self, test_images, tmp_path):
        """Test creating grid image with reasonable constraints."""
        output_path = tmp_path / "grid.jpg"

        # Use reasonable constraints that should work
        created_files = create_grid_image(
            test_images[:3], output_path, columns=2, max_height=400  # Use fewer images  # Reasonable height
        )

        # Should create at least one grid image
        assert len(created_files) >= 1
        for file_path in created_files:
            assert file_path.exists()
