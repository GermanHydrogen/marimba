"""Tests for marimba.lib.video module."""

import pytest

from marimba.lib.video import (
    filter_existing_thumbnails,
    generate_potential_filenames,
    generate_video_thumbnails,
    get_stream_properties,
    save_thumbnail,
)


class TestVideoUtilities:
    """Test video utility functions."""

    @pytest.fixture
    def mock_video_stream(self, mocker):
        """Create a mock video stream."""
        stream = mocker.Mock()
        stream.average_rate = 30.0
        stream.time_base = 1 / 30.0
        stream.frames = 900
        return stream

    @pytest.fixture
    def test_video_path(self, tmp_path):
        """Create a test video path."""
        return tmp_path / "test_video.mp4"

    @pytest.mark.integration
    def test_get_stream_properties(self, mock_video_stream):
        """Test extracting stream properties."""
        frame_rate, time_base, total_frames = get_stream_properties(mock_video_stream)

        assert frame_rate == 30.0
        assert time_base == 1 / 30.0
        assert total_frames == 900

    @pytest.mark.integration
    def test_get_stream_properties_none_frame_rate(self, mocker):
        """Test get_stream_properties with None frame rate."""
        stream = mocker.Mock()
        stream.average_rate = None
        stream.time_base = 1 / 30.0
        stream.frames = 900

        with pytest.raises(ValueError, match="Frame rate or time base is None"):
            get_stream_properties(stream)

    @pytest.mark.integration
    def test_get_stream_properties_none_time_base(self, mocker):
        """Test get_stream_properties with None time base."""
        stream = mocker.Mock()
        stream.average_rate = 30.0
        stream.time_base = None
        stream.frames = 900

        with pytest.raises(ValueError, match="Frame rate or time base is None"):
            get_stream_properties(stream)

    @pytest.mark.integration
    def test_generate_potential_filenames(self, test_video_path, tmp_path):
        """Test generating potential filenames."""
        output_dir = tmp_path / "output"

        filenames = generate_potential_filenames(test_video_path, output_dir, 100, 10, "_THUMB")

        assert len(filenames) == 10  # 100 frames / 10 interval
        assert 0 in filenames
        assert 90 in filenames

        # Check filename format
        expected_filename = "test_video_000_THUMB.JPG"
        assert filenames[0].name == expected_filename

    @pytest.mark.integration
    def test_generate_potential_filenames_padding(self, test_video_path, tmp_path):
        """Test filename padding with different frame counts."""
        output_dir = tmp_path / "output"

        # Test with 1000 frames (4-digit padding)
        filenames = generate_potential_filenames(test_video_path, output_dir, 1000, 100, "_THUMB")

        expected_filename = "test_video_0000_THUMB.JPG"
        assert filenames[0].name == expected_filename

    @pytest.mark.integration
    def test_filter_existing_thumbnails_no_overwrite(self, tmp_path):
        """Test filtering existing thumbnails without overwrite."""
        # Create some existing files
        existing_file1 = tmp_path / "file1.jpg"
        existing_file2 = tmp_path / "file2.jpg"
        non_existing_file = tmp_path / "file3.jpg"

        existing_file1.touch()
        existing_file2.touch()

        potential_filenames = {
            0: existing_file1,
            1: existing_file2,
            2: non_existing_file,
        }

        existing_paths = filter_existing_thumbnails(potential_filenames, overwrite=False)

        assert len(existing_paths) == 2
        assert existing_file1 in existing_paths
        assert existing_file2 in existing_paths

        # Should only have non-existing file remaining
        assert len(potential_filenames) == 1
        assert 2 in potential_filenames

    @pytest.mark.integration
    def test_filter_existing_thumbnails_with_overwrite(self, tmp_path):
        """Test filtering existing thumbnails with overwrite enabled."""
        existing_file = tmp_path / "file1.jpg"
        existing_file.touch()

        potential_filenames = {
            0: existing_file,
        }

        existing_paths = filter_existing_thumbnails(potential_filenames, overwrite=True)

        assert len(existing_paths) == 0
        # Should still have all files when overwrite is True
        assert len(potential_filenames) == 1

    @pytest.mark.integration
    def test_filter_existing_thumbnails_logging(self, mocker, tmp_path):
        """Test that filtering logs existing files."""
        mock_logger = mocker.patch("marimba.lib.video.logger")
        existing_file = tmp_path / "file1.jpg"
        existing_file.touch()

        potential_filenames = {0: existing_file}

        filter_existing_thumbnails(potential_filenames, overwrite=False)

        mock_logger.info.assert_called_once()
        assert "Thumbnail already exists" in str(mock_logger.info.call_args)

    @pytest.mark.integration
    def test_save_thumbnail(self, mocker, tmp_path):
        """Test saving thumbnail from video frame."""
        output_path = tmp_path / "thumb.jpg"

        # Mock the video frame and image
        mock_frame = mocker.Mock()
        mock_image = mocker.Mock()
        mock_frame.to_image.return_value = mock_image

        save_thumbnail(mock_frame, output_path)

        mock_frame.to_image.assert_called_once()
        # Just check that thumbnail and save were called
        mock_image.thumbnail.assert_called_once()
        mock_image.save.assert_called_once_with(output_path)

    @pytest.mark.integration
    def test_generate_video_thumbnails_basic(
        self,
        mocker,
        test_video_path,
        tmp_path,
    ):
        """Test basic video thumbnail generation."""
        output_dir = tmp_path / "output"

        mock_av_open = mocker.patch("av.open")
        mock_get_properties = mocker.patch("marimba.lib.video.get_stream_properties")
        mock_generate_filenames = mocker.patch("marimba.lib.video.generate_potential_filenames")
        mock_filter = mocker.patch("marimba.lib.video.filter_existing_thumbnails")

        # Mock the container and stream
        mock_container = mocker.Mock()
        mock_stream = mocker.Mock()
        mock_container.streams.video = [mock_stream]
        mock_av_open.return_value = mock_container

        # Mock the stream properties
        mock_get_properties.return_value = (30.0, 1 / 30.0, 900)

        # Mock potential filenames (empty to skip processing)
        mock_generate_filenames.return_value = {}

        # Mock existing thumbnails
        mock_filter.return_value = []

        result_video, result_paths = generate_video_thumbnails(test_video_path, output_dir)

        assert result_video == test_video_path
        assert result_paths == []
        assert output_dir.exists()

    @pytest.mark.integration
    def test_generate_video_thumbnails_with_frames(
        self,
        mocker,
        test_video_path,
        tmp_path,
    ):
        """Test video thumbnail generation with no potential filenames (simpler case)."""
        output_dir = tmp_path / "output"

        mock_av_open = mocker.patch("av.open")
        mock_get_properties = mocker.patch("marimba.lib.video.get_stream_properties")
        mock_generate_filenames = mocker.patch("marimba.lib.video.generate_potential_filenames")
        mock_filter = mocker.patch("marimba.lib.video.filter_existing_thumbnails")

        # Mock the container and stream
        mock_container = mocker.Mock()
        mock_stream = mocker.Mock()
        mock_container.streams.video = [mock_stream]
        mock_av_open.return_value = mock_container

        # Mock the stream properties
        mock_get_properties.return_value = (30.0, 1 / 30.0, 900)

        # No potential filenames - should skip frame processing
        mock_generate_filenames.return_value = {}

        # Mock existing thumbnails
        existing_paths = [tmp_path / "existing.jpg"]
        mock_filter.return_value = existing_paths

        result_video, result_paths = generate_video_thumbnails(test_video_path, output_dir)

        assert result_video == test_video_path
        assert result_paths == existing_paths

    @pytest.mark.integration
    def test_generate_video_thumbnails_av_error(self, mocker, test_video_path, tmp_path):
        """Test video thumbnail generation with AV error."""
        output_dir = tmp_path / "output"

        mock_av_open = mocker.patch("av.open")
        mock_av_open.side_effect = Exception("Generic AV error")

        with pytest.raises(Exception, match="Generic AV error"):
            generate_video_thumbnails(test_video_path, output_dir)

    @pytest.mark.integration
    def test_generate_video_thumbnails_ffmpeg_error(self, mocker, test_video_path, tmp_path):
        """Test video thumbnail generation with FFmpeg dependency error."""
        output_dir = tmp_path / "output"

        mock_av_open = mocker.patch("av.open")
        mock_show_error = mocker.patch("marimba.lib.video.show_dependency_error_and_exit")
        mock_av_open.side_effect = Exception("No such file or directory: ffmpeg not found")

        try:
            generate_video_thumbnails(test_video_path, output_dir)
        except Exception:
            pass  # We expect this to call show_dependency_error_and_exit

        mock_show_error.assert_called_once()

    @pytest.mark.integration
    def test_generate_video_thumbnails_custom_params(
        self,
        mocker,
        test_video_path,
        tmp_path,
    ):
        """Test video thumbnail generation with custom parameters."""
        output_dir = tmp_path / "output"

        mock_av_open = mocker.patch("av.open")
        mock_get_properties = mocker.patch("marimba.lib.video.get_stream_properties")
        mock_generate_filenames = mocker.patch("marimba.lib.video.generate_potential_filenames")
        mock_filter = mocker.patch("marimba.lib.video.filter_existing_thumbnails")

        # Mock the container and stream
        mock_container = mocker.Mock()
        mock_stream = mocker.Mock()
        mock_container.streams.video = [mock_stream]
        mock_av_open.return_value = mock_container

        # Mock the stream properties
        mock_get_properties.return_value = (25.0, 1 / 25.0, 750)

        # Mock potential filenames (empty to skip processing)
        mock_generate_filenames.return_value = {}

        # Mock existing thumbnails
        mock_filter.return_value = []

        result_video, result_paths = generate_video_thumbnails(
            test_video_path,
            output_dir,
            interval=5,
            suffix="_CUSTOM",
            overwrite=True,
        )

        # Check that custom parameters were passed through
        mock_generate_filenames.assert_called_once_with(
            test_video_path,
            output_dir,
            750,
            125,
            "_CUSTOM",  # 25 fps * 5 seconds = 125 frame interval
        )
        mock_filter.assert_called_once_with({}, True)  # overwrite=True

        assert result_video == test_video_path

    @pytest.mark.integration
    def test_generate_video_thumbnails_with_frame_processing(
        self,
        mocker,
        test_video_path,
        tmp_path,
    ):
        """Test video thumbnail generation with actual frame processing."""
        output_dir = tmp_path / "output"

        mock_av_open = mocker.patch("av.open")
        mock_get_properties = mocker.patch("marimba.lib.video.get_stream_properties")
        mock_generate_filenames = mocker.patch("marimba.lib.video.generate_potential_filenames")
        mock_filter = mocker.patch("marimba.lib.video.filter_existing_thumbnails")
        mock_save_thumbnail = mocker.patch("marimba.lib.video.save_thumbnail")
        mock_logger = mocker.patch("marimba.lib.video.logger")

        # Create mock video frame with proper type
        import av

        mock_frame = mocker.Mock(spec=av.video.frame.VideoFrame)
        mock_frame.pts = 10

        # Create mock packet that yields frames
        mock_packet = mocker.Mock()
        mock_packet.decode.return_value = [mock_frame]

        # Create mock container with demux that yields packets
        mock_container = mocker.Mock()
        mock_stream = mocker.Mock()
        mock_container.streams.video = [mock_stream]
        mock_container.demux.return_value = [mock_packet]
        mock_av_open.return_value = mock_container

        # Mock the stream properties
        mock_get_properties.return_value = (30.0, 1 / 30.0, 900)  # 30fps, time_base, 900 frames

        # Mock potential filenames to match calculated frame number
        # frame_number = int(frame.pts * time_base * frame_rate) = int(10 * (1/30) * 30) = 10
        expected_frame_number = 10
        output_path = output_dir / "test_video_000010_THUMB.JPG"
        mock_generate_filenames.return_value = {expected_frame_number: output_path}

        # Mock existing thumbnails (empty)
        mock_filter.return_value = []

        result_video, result_paths = generate_video_thumbnails(test_video_path, output_dir)

        # Verify frame processing happened
        mock_container.demux.assert_called_once_with(mock_stream)
        mock_packet.decode.assert_called_once()

        # Verify thumbnail generation
        mock_save_thumbnail.assert_called_once_with(mock_frame, output_path)
        mock_logger.info.assert_called_once()

        assert result_video == test_video_path
        assert len(result_paths) == 1
        assert output_path in result_paths

    @pytest.mark.integration
    def test_generate_video_thumbnails_early_exit_no_overwrite(
        self,
        mocker,
        test_video_path,
        tmp_path,
    ):
        """Test early exit when not overwriting and no potential filenames left."""
        output_dir = tmp_path / "output"

        mock_av_open = mocker.patch("av.open")
        mock_get_properties = mocker.patch("marimba.lib.video.get_stream_properties")
        mock_generate_filenames = mocker.patch("marimba.lib.video.generate_potential_filenames")
        mock_filter = mocker.patch("marimba.lib.video.filter_existing_thumbnails")
        mock_save_thumbnail = mocker.patch("marimba.lib.video.save_thumbnail")

        # Create mock video frame with proper type
        import av

        mock_frame = mocker.Mock(spec=av.video.frame.VideoFrame)
        mock_frame.pts = 10

        # Create mock packet that yields frames
        mock_packet = mocker.Mock()
        mock_packet.decode.return_value = [mock_frame]

        # Create mock container
        mock_container = mocker.Mock()
        mock_stream = mocker.Mock()
        mock_container.streams.video = [mock_stream]
        mock_container.demux.return_value = [mock_packet]
        mock_av_open.return_value = mock_container

        # Mock the stream properties
        mock_get_properties.return_value = (30.0, 1 / 30.0, 900)

        # Mock potential filenames - will be consumed during processing
        expected_frame_number = 10
        output_path = output_dir / "test_video_000010_THUMB.JPG"
        mock_generate_filenames.return_value = {expected_frame_number: output_path}

        # Mock existing thumbnails
        mock_filter.return_value = []

        # Call with overwrite=False (default)
        result_video, result_paths = generate_video_thumbnails(test_video_path, output_dir, overwrite=False)

        # Should process one frame and then exit early because potential_filenames becomes empty
        mock_save_thumbnail.assert_called_once_with(mock_frame, output_path)

        assert result_video == test_video_path
        assert len(result_paths) == 1
