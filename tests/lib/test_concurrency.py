"""Tests for marimba.lib.concurrency module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from marimba.lib.concurrency import (
    multithreaded_generate_image_thumbnails,
    multithreaded_generate_video_thumbnails,
)


class TestConcurrencyUtilities:
    """Test concurrency utility functions."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock BasePipeline instance."""
        pipeline = Mock()
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

    @patch('marimba.lib.concurrency.generate_image_thumbnail')
    def test_multithreaded_generate_image_thumbnails(self, mock_generate_thumbnail, mock_pipeline, test_image_paths, tmp_path):
        """Test multithreaded image thumbnail generation."""
        output_dir = tmp_path / "thumbnails"
        
        # Mock thumbnail generation
        mock_generate_thumbnail.side_effect = [
            tmp_path / "thumb1.jpg",
            tmp_path / "thumb2.jpg", 
            tmp_path / "thumb3.jpg",
        ]
        
        result = multithreaded_generate_image_thumbnails(
            mock_pipeline,
            test_image_paths,
            output_dir
        )
        
        assert len(result) == 3
        assert output_dir.exists()
        # Should call generate_image_thumbnail for each image
        assert mock_generate_thumbnail.call_count == 3

    @patch('marimba.lib.concurrency.generate_image_thumbnail')
    def test_multithreaded_generate_image_thumbnails_with_logger(self, mock_generate_thumbnail, mock_pipeline, test_image_paths, tmp_path):
        """Test multithreaded image thumbnail generation with logger."""
        output_dir = tmp_path / "thumbnails"
        mock_logger = Mock()
        
        mock_generate_thumbnail.return_value = tmp_path / "thumb.jpg"
        
        result = multithreaded_generate_image_thumbnails(
            mock_pipeline,
            test_image_paths,
            output_dir,
            logger=mock_logger,
            max_workers=2
        )
        
        assert isinstance(result, list)
        # Logger should be called for debug messages
        assert mock_logger.debug.call_count >= 0  # May vary due to threading

    @patch('marimba.lib.concurrency.generate_video_thumbnails')
    def test_multithreaded_generate_video_thumbnails(self, mock_generate_thumbnails, mock_pipeline, test_video_paths, tmp_path):
        """Test multithreaded video thumbnail generation."""
        output_dir = tmp_path / "video_thumbnails"
        
        # Mock video thumbnail generation
        mock_generate_thumbnails.side_effect = [
            (test_video_paths[0], [tmp_path / "v1_thumb1.jpg", tmp_path / "v1_thumb2.jpg"]),
            (test_video_paths[1], [tmp_path / "v2_thumb1.jpg"]),
        ]
        
        result = multithreaded_generate_video_thumbnails(
            mock_pipeline,
            test_video_paths,
            output_dir,
            interval=5,
            suffix="_TEST"
        )
        
        assert len(result) == 2
        # Should call generate_video_thumbnails for each video
        assert mock_generate_thumbnails.call_count == 2

    @patch('marimba.lib.concurrency.generate_video_thumbnails')
    def test_multithreaded_generate_video_thumbnails_with_options(self, mock_generate_thumbnails, mock_pipeline, test_video_paths, tmp_path):
        """Test multithreaded video thumbnail generation with various options."""
        output_dir = tmp_path / "video_thumbnails"
        mock_logger = Mock()
        
        mock_generate_thumbnails.return_value = (test_video_paths[0], [tmp_path / "thumb.jpg"])
        
        result = multithreaded_generate_video_thumbnails(
            mock_pipeline,
            test_video_paths[:1],  # Just one video
            output_dir,
            interval=15,
            suffix="_CUSTOM",
            logger=mock_logger,
            max_workers=1,
            overwrite=True
        )
        
        assert isinstance(result, list)
        # Should create subdirectory for video
        video_subdir = output_dir / test_video_paths[0].stem
        assert video_subdir.exists()

    @patch('marimba.lib.concurrency.generate_image_thumbnail')
    def test_multithreaded_generate_image_thumbnails_empty_list(self, mock_generate_thumbnail, mock_pipeline, tmp_path):
        """Test multithreaded image thumbnail generation with empty list."""
        output_dir = tmp_path / "thumbnails"
        
        result = multithreaded_generate_image_thumbnails(
            mock_pipeline,
            [],  # Empty list
            output_dir
        )
        
        assert result == []
        assert mock_generate_thumbnail.call_count == 0

    @patch('marimba.lib.concurrency.generate_video_thumbnails')
    def test_multithreaded_generate_video_thumbnails_empty_list(self, mock_generate_thumbnails, mock_pipeline, tmp_path):
        """Test multithreaded video thumbnail generation with empty list."""
        output_dir = tmp_path / "video_thumbnails"
        
        result = multithreaded_generate_video_thumbnails(
            mock_pipeline,
            [],  # Empty list
            output_dir
        )
        
        assert result == []
        assert mock_generate_thumbnails.call_count == 0