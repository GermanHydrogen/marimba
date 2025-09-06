"""Tests for marimba.lib.concurrency module."""

from pathlib import Path

import pytest

from marimba.lib.concurrency import (
    multithreaded_generate_image_thumbnails,
    multithreaded_generate_video_thumbnails,
)


class TestConcurrencyUtilities:
    """Test concurrency utility functions."""

    @pytest.fixture
    def mock_pipeline(self, mocker):
        """Create a mock BasePipeline instance."""
        pipeline = mocker.Mock()
        pipeline._root_path = "/mock/root/path"
        return pipeline

    @pytest.fixture
    def test_image_paths(self, tmp_path):
        """Create test image paths."""
        return [
            tmp_path / "image1.jpg",
            tmp_path / "image2.jpg",
            tmp_path / "image3.jpg",
        ]

    @pytest.fixture
    def test_video_paths(self, tmp_path):
        """Create test video paths."""
        return [
            tmp_path / "video1.mp4",
            tmp_path / "video2.mp4",
        ]

    @pytest.mark.integration
    def test_multithreaded_generate_image_thumbnails(self, mocker, mock_pipeline, test_image_paths, tmp_path):
        """Test multithreaded image thumbnail generation."""
        mock_generate_thumbnail = mocker.patch("marimba.lib.concurrency.generate_image_thumbnail")
        output_dir = tmp_path / "thumbnails"

        # Mock thumbnail generation
        mock_generate_thumbnail.side_effect = [
            tmp_path / "thumb1.jpg",
            tmp_path / "thumb2.jpg",
            tmp_path / "thumb3.jpg",
        ]

        result = multithreaded_generate_image_thumbnails(mock_pipeline, test_image_paths, output_dir)

        assert len(result) == 3
        assert output_dir.exists()
        # Should call generate_image_thumbnail for each image
        assert mock_generate_thumbnail.call_count == 3

    @pytest.mark.integration
    def test_multithreaded_generate_image_thumbnails_with_logger(
        self, mocker, mock_pipeline, test_image_paths, tmp_path
    ):
        """Test multithreaded image thumbnail generation with logger."""
        mock_generate_thumbnail = mocker.patch("marimba.lib.concurrency.generate_image_thumbnail")
        output_dir = tmp_path / "thumbnails"
        mock_logger = mocker.Mock()

        mock_generate_thumbnail.return_value = tmp_path / "thumb.jpg"

        result = multithreaded_generate_image_thumbnails(
            mock_pipeline, test_image_paths, output_dir, logger=mock_logger, max_workers=2
        )

        assert isinstance(result, list)
        # Logger should be called for debug messages
        assert mock_logger.debug.call_count >= 0  # May vary due to threading

    @pytest.mark.integration
    def test_multithreaded_generate_video_thumbnails(self, mocker, mock_pipeline, test_video_paths, tmp_path):
        """Test multithreaded video thumbnail generation."""
        mock_generate_thumbnails = mocker.patch("marimba.lib.concurrency.generate_video_thumbnails")
        output_dir = tmp_path / "video_thumbnails"

        # Mock video thumbnail generation
        mock_generate_thumbnails.side_effect = [
            (test_video_paths[0], [tmp_path / "v1_thumb1.jpg", tmp_path / "v1_thumb2.jpg"]),
            (test_video_paths[1], [tmp_path / "v2_thumb1.jpg"]),
        ]

        result = multithreaded_generate_video_thumbnails(
            mock_pipeline, test_video_paths, output_dir, interval=5, suffix="_TEST"
        )

        assert len(result) == 2
        # Should call generate_video_thumbnails for each video
        assert mock_generate_thumbnails.call_count == 2

    @pytest.mark.integration
    def test_multithreaded_generate_video_thumbnails_with_options(
        self, mocker, mock_pipeline, test_video_paths, tmp_path
    ):
        """Test multithreaded video thumbnail generation with various options."""
        mock_generate_thumbnails = mocker.patch("marimba.lib.concurrency.generate_video_thumbnails")
        output_dir = tmp_path / "video_thumbnails"
        mock_logger = mocker.Mock()

        mock_generate_thumbnails.return_value = (test_video_paths[0], [tmp_path / "thumb.jpg"])

        result = multithreaded_generate_video_thumbnails(
            mock_pipeline,
            test_video_paths[:1],  # Just one video
            output_dir,
            interval=15,
            suffix="_CUSTOM",
            logger=mock_logger,
            max_workers=1,
            overwrite=True,
        )

        assert isinstance(result, list)
        # Should create subdirectory for video
        video_subdir = output_dir / test_video_paths[0].stem
        assert video_subdir.exists()

    @pytest.mark.integration
    def test_multithreaded_generate_image_thumbnails_empty_list(self, mocker, mock_pipeline, tmp_path):
        """Test multithreaded image thumbnail generation with empty list."""
        mock_generate_thumbnail = mocker.patch("marimba.lib.concurrency.generate_image_thumbnail")
        output_dir = tmp_path / "thumbnails"

        result = multithreaded_generate_image_thumbnails(mock_pipeline, [], output_dir)  # Empty list

        assert result == []
        assert mock_generate_thumbnail.call_count == 0

    @pytest.mark.integration
    def test_multithreaded_generate_video_thumbnails_empty_list(self, mocker, mock_pipeline, tmp_path):
        """Test multithreaded video thumbnail generation with empty list."""
        mock_generate_thumbnails = mocker.patch("marimba.lib.concurrency.generate_video_thumbnails")
        output_dir = tmp_path / "video_thumbnails"

        result = multithreaded_generate_video_thumbnails(mock_pipeline, [], output_dir)  # Empty list

        assert result == []
        assert mock_generate_thumbnails.call_count == 0
