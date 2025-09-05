from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch, Mock

import pytest
from ifdo.models import ImageData

from marimba.core.schemas.base import BaseMetadata
from marimba.core.schemas.ifdo import iFDOMetadata


class TestiFDOMetadataProperties:
    """Test all properties of iFDOMetadata class."""

    @pytest.fixture
    def sample_image_data(self) -> ImageData:
        """Create a sample ImageData for testing."""
        return ImageData(
            image_datetime=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            image_latitude=45.0,
            image_longitude=-123.0,
            image_altitude_meters=100.0,
            image_hash_sha256="abc123",
        )

    @pytest.fixture
    def ifdo_metadata(self, sample_image_data: ImageData) -> iFDOMetadata:
        """Create iFDOMetadata instance for testing."""
        return iFDOMetadata(sample_image_data)

    @pytest.mark.unit
    def test_datetime_property(self, ifdo_metadata: iFDOMetadata) -> None:
        """Test datetime property returns correct value."""
        assert ifdo_metadata.datetime == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.mark.unit
    def test_datetime_property_none(self):
        """Test datetime property when None."""
        image_data = ImageData(image_datetime=None)
        metadata = iFDOMetadata(image_data)
        assert metadata.datetime is None

    @pytest.mark.unit
    def test_datetime_property_wrong_type(self):
        """Test datetime property raises TypeError for wrong type."""
        image_data = ImageData()
        image_data.image_datetime = "not a datetime"  # Force wrong type
        metadata = iFDOMetadata(image_data)

        with pytest.raises(TypeError, match="Expected datetime or None"):
            _ = metadata.datetime

    @pytest.mark.unit
    def test_latitude_property(self, ifdo_metadata: iFDOMetadata) -> None:
        """Test latitude property returns correct value."""
        assert ifdo_metadata.latitude == 45.0

    @pytest.mark.unit
    def test_latitude_property_none(self):
        """Test latitude property when None."""
        image_data = ImageData(image_latitude=None)
        metadata = iFDOMetadata(image_data)
        assert metadata.latitude is None

    @pytest.mark.unit
    def test_latitude_property_int(self):
        """Test latitude property accepts integer values."""
        image_data = ImageData(image_latitude=45)
        metadata = iFDOMetadata(image_data)
        assert metadata.latitude == 45.0

    @pytest.mark.unit
    def test_latitude_property_wrong_type(self):
        """Test latitude property raises TypeError for wrong type."""
        image_data = ImageData()
        image_data.image_latitude = "not a number"  # Force wrong type
        metadata = iFDOMetadata(image_data)

        with pytest.raises(TypeError, match="Expected float or None"):
            _ = metadata.latitude

    @pytest.mark.unit
    def test_longitude_property(self, ifdo_metadata: iFDOMetadata) -> None:
        """Test longitude property returns correct value."""
        assert ifdo_metadata.longitude == -123.0

    @pytest.mark.unit
    def test_longitude_property_none(self):
        """Test longitude property when None."""
        image_data = ImageData(image_longitude=None)
        metadata = iFDOMetadata(image_data)
        assert metadata.longitude is None

    @pytest.mark.unit
    def test_longitude_property_int(self):
        """Test longitude property accepts integer values."""
        image_data = ImageData(image_longitude=-123)
        metadata = iFDOMetadata(image_data)
        assert metadata.longitude == -123.0

    @pytest.mark.unit
    def test_longitude_property_wrong_type(self):
        """Test longitude property raises TypeError for wrong type."""
        image_data = ImageData()
        image_data.image_longitude = "not a number"  # Force wrong type
        metadata = iFDOMetadata(image_data)

        with pytest.raises(TypeError, match="Expected float or None"):
            _ = metadata.longitude

    @pytest.mark.unit
    def test_altitude_property(self, ifdo_metadata: iFDOMetadata) -> None:
        """Test altitude property returns correct value."""
        assert ifdo_metadata.altitude == 100.0

    @pytest.mark.unit
    def test_altitude_property_none(self):
        """Test altitude property when None."""
        image_data = ImageData(image_altitude_meters=None)
        metadata = iFDOMetadata(image_data)
        assert metadata.altitude is None

    @pytest.mark.unit
    def test_altitude_property_int(self):
        """Test altitude property accepts integer values."""
        image_data = ImageData(image_altitude_meters=100)
        metadata = iFDOMetadata(image_data)
        assert metadata.altitude == 100.0

    @pytest.mark.unit
    def test_altitude_property_wrong_type(self):
        """Test altitude property raises TypeError for wrong type."""
        image_data = ImageData()
        image_data.image_altitude_meters = "not a number"  # Force wrong type
        metadata = iFDOMetadata(image_data)

        with pytest.raises(TypeError, match="Expected float or None"):
            _ = metadata.altitude

    @pytest.mark.unit
    def test_context_property(self):
        """Test context property returns correct value."""
        mock_context = Mock()
        mock_context.name = "test context"
        image_data = ImageData()
        image_data.image_context = mock_context
        metadata = iFDOMetadata(image_data)

        assert metadata.context == "test context"

    @pytest.mark.unit
    def test_context_property_none(self):
        """Test context property when None."""
        image_data = ImageData(image_context=None)
        metadata = iFDOMetadata(image_data)
        assert metadata.context is None

    @pytest.mark.unit
    def test_license_property(self):
        """Test license property returns correct value."""
        mock_license = Mock()
        mock_license.name = "MIT License"
        image_data = ImageData()
        image_data.image_license = mock_license
        metadata = iFDOMetadata(image_data)

        assert metadata.license == "MIT License"

    @pytest.mark.unit
    def test_license_property_none(self):
        """Test license property when None."""
        image_data = ImageData(image_license=None)
        metadata = iFDOMetadata(image_data)
        assert metadata.license is None

    @pytest.mark.unit
    def test_creators_property(self):
        """Test creators property returns correct list."""
        mock_creator1 = Mock()
        mock_creator1.name = "Creator One"
        mock_creator2 = Mock()
        mock_creator2.name = "Creator Two"

        image_data = ImageData()
        image_data.image_creators = [mock_creator1, mock_creator2]
        metadata = iFDOMetadata(image_data)

        assert metadata.creators == ["Creator One", "Creator Two"]

    @pytest.mark.unit
    def test_creators_property_empty(self):
        """Test creators property when empty."""
        image_data = ImageData(image_creators=[])
        metadata = iFDOMetadata(image_data)
        assert metadata.creators == []

    @pytest.mark.unit
    def test_creators_property_none(self):
        """Test creators property when None."""
        image_data = ImageData(image_creators=None)
        metadata = iFDOMetadata(image_data)
        assert metadata.creators == []

    @pytest.mark.unit
    def test_hash_sha256_property(self, ifdo_metadata: iFDOMetadata) -> None:
        """Test hash_sha256 property returns correct value."""
        assert ifdo_metadata.hash_sha256 == "abc123"

    @pytest.mark.unit
    def test_hash_sha256_property_none(self):
        """Test hash_sha256 property when None."""
        image_data = ImageData(image_hash_sha256=None)
        metadata = iFDOMetadata(image_data)
        assert metadata.hash_sha256 is None

    @pytest.mark.unit
    def test_hash_sha256_property_wrong_type(self):
        """Test hash_sha256 property raises TypeError for wrong type."""
        image_data = ImageData()
        image_data.image_hash_sha256 = 123  # Force wrong type
        metadata = iFDOMetadata(image_data)

        with pytest.raises(TypeError, match="Expected str or None"):
            _ = metadata.hash_sha256

    @pytest.mark.unit
    def test_hash_sha256_setter(self, ifdo_metadata: iFDOMetadata) -> None:
        """Test hash_sha256 setter works correctly."""
        ifdo_metadata.hash_sha256 = "new_hash"
        assert ifdo_metadata.hash_sha256 == "new_hash"

    @pytest.mark.unit
    def test_is_video_property_false(self, ifdo_metadata: iFDOMetadata) -> None:
        """Test is_video property returns False for single ImageData."""
        assert ifdo_metadata.is_video is False

    @pytest.mark.unit
    def test_is_video_property_true(self):
        """Test is_video property returns True for list of ImageData."""
        image_data_list = [ImageData(), ImageData()]
        metadata = iFDOMetadata(image_data_list)
        assert metadata.is_video is True

    @pytest.mark.unit
    def test_primary_image_data_single(self, sample_image_data: ImageData) -> None:
        """Test primary_image_data for single ImageData."""
        metadata = iFDOMetadata(sample_image_data)
        assert metadata.primary_image_data is sample_image_data

    @pytest.mark.unit
    def test_primary_image_data_list(self):
        """Test primary_image_data for list of ImageData."""
        image_data1 = ImageData(image_altitude_meters=100.0)
        image_data2 = ImageData(image_altitude_meters=200.0)
        metadata = iFDOMetadata([image_data1, image_data2])
        assert metadata.primary_image_data is image_data1


class TestiFDOMetadataStaticMethods:
    """Test static methods of iFDOMetadata class."""

    @pytest.mark.unit
    def test_is_video_file_video_extensions(self):
        """Test _is_video_file returns True for video extensions."""
        video_files = [
            "movie.mp4",
            "video.avi",
            "clip.mov",
            "film.wmv",
            "animation.webm",
            "test.mkv",
            "sample.m4v",
            "mobile.3gp",
            "open.ogv",
            "stream.ts",
        ]

        for filename in video_files:
            assert iFDOMetadata._is_video_file(filename) is True

    @pytest.mark.unit
    def test_is_video_file_image_extensions(self):
        """Test _is_video_file returns False for image extensions."""
        image_files = ["photo.jpg", "image.png", "graphic.bmp", "picture.tiff", "icon.gif"]

        for filename in image_files:
            assert iFDOMetadata._is_video_file(filename) is False

    @pytest.mark.unit
    def test_is_video_file_case_insensitive(self):
        """Test _is_video_file is case insensitive."""
        assert iFDOMetadata._is_video_file("VIDEO.MP4") is True
        assert iFDOMetadata._is_video_file("video.MP4") is True
        assert iFDOMetadata._is_video_file("VIDEO.mp4") is True


class TestiFDOMetadataProcessMethods:
    """Test processing methods of iFDOMetadata class."""

    @pytest.mark.unit
    def test_process_video_metadata(self):
        """Test _process_video_metadata method."""
        # Create test data
        image_data1 = ImageData(image_altitude_meters=100.0)
        image_data2 = ImageData(image_altitude_meters=200.0)
        image_data3 = ImageData(image_altitude_meters=300.0)

        # Mix of single and list metadata
        single_metadata = iFDOMetadata(image_data1)
        list_metadata = iFDOMetadata([image_data2, image_data3])

        path = Path("subdir/video.mp4")
        result = iFDOMetadata._process_video_metadata([single_metadata, list_metadata], path)

        assert len(result) == 3
        assert result[0].image_altitude_meters == 100.0
        assert result[1].image_altitude_meters == 200.0
        assert result[2].image_altitude_meters == 300.0

        # Check that image_set_local_path is set for subdirectory
        for img_data in result:
            assert img_data.image_set_local_path == "subdir"

    @pytest.mark.unit
    def test_process_image_metadata(self):
        """Test _process_image_metadata method."""
        image_data = ImageData(image_altitude_meters=100.0)
        metadata = iFDOMetadata(image_data)

        path = Path("subdir/image.jpg")
        result = iFDOMetadata._process_image_metadata([metadata], path)

        assert result.image_altitude_meters == 100.0
        assert result.image_set_local_path == "subdir"

    @pytest.mark.unit
    def test_process_image_metadata_root_path(self):
        """Test _process_image_metadata with root path doesn't set local_path."""
        image_data = ImageData(image_altitude_meters=100.0)
        metadata = iFDOMetadata(image_data)

        path = Path("image.jpg")  # Root level
        result = iFDOMetadata._process_image_metadata([metadata], path)

        assert result.image_altitude_meters == 100.0
        assert not hasattr(result, "image_set_local_path") or result.image_set_local_path is None


@pytest.mark.integration
def test_create_dataset_metadata():
    mock_uuid = "a43a84f2-b657-44e0-bafe-72e2624115fa"

    def mock_saver(path: Path, output_name: str, data: dict[str, Any]) -> None:
        assert path.name == "tmp"
        assert output_name == "ifdo"
        assert data == {
            "image-set-header": {
                "image-set-name": "TestDataSet",
                "image-set-uuid": mock_uuid,
                "image-set-handle": "",
                "image-set-ifdo-version": "v2.1.0",
            },
            "image-set-items": {"image.jpg": {"image-altitude-meters": 0.0}},
        }

    data_setname = "TestDataSet"
    root_dir = Path("/tmp")
    items = {"image.jpg": [cast(BaseMetadata, iFDOMetadata(image_data=ImageData(image_altitude_meters=0.0)))]}
    with patch("uuid.uuid4", MagicMock(return_value=mock_uuid)):
        iFDOMetadata.create_dataset_metadata(data_setname, root_dir, items, saver_overwrite=mock_saver)


@pytest.mark.integration
def test_create_dataset_metadata_with_metadata_name():
    """Test create_dataset_metadata with custom metadata name."""
    mock_uuid = "test-uuid"

    def mock_saver(path: Path, output_name: str, data: dict[str, Any]) -> None:
        assert output_name == "custom.ifdo"

    image_data = ImageData(image_altitude_meters=0.0)
    items = {"image.jpg": [cast(BaseMetadata, iFDOMetadata(image_data))]}

    with patch("uuid.uuid4", MagicMock(return_value=mock_uuid)):
        iFDOMetadata.create_dataset_metadata(
            "TestDataSet", Path("/tmp"), items, metadata_name="custom", saver_overwrite=mock_saver
        )


@pytest.mark.integration
def test_create_dataset_metadata_video_file():
    """Test create_dataset_metadata handles video files correctly."""
    mock_uuid = "test-uuid"

    def mock_saver(path: Path, output_name: str, data: dict[str, Any]) -> None:
        # Video files should create lists in image-set-items
        assert "video.mp4" in data["image-set-items"]
        assert isinstance(data["image-set-items"]["video.mp4"], list)

    # Create video metadata (list of ImageData)
    image_data_list = [ImageData(image_altitude_meters=100.0), ImageData(image_altitude_meters=200.0)]
    video_metadata = iFDOMetadata(image_data_list)
    items = {"video.mp4": [cast(BaseMetadata, video_metadata)]}

    with patch("uuid.uuid4", MagicMock(return_value=mock_uuid)):
        iFDOMetadata.create_dataset_metadata("TestDataSet", Path("/tmp"), items, saver_overwrite=mock_saver)


@pytest.mark.integration
def test_create_dataset_metadata_dry_run():
    """Test create_dataset_metadata with dry_run=True doesn't call saver."""

    def mock_saver(path: Path, output_name: str, data: dict[str, Any]) -> None:
        pytest.fail("Saver should not be called in dry run mode")

    image_data = ImageData(image_altitude_meters=0.0)
    items = {"image.jpg": [cast(BaseMetadata, iFDOMetadata(image_data))]}

    # Should not call saver
    iFDOMetadata.create_dataset_metadata("TestDataSet", Path("/tmp"), items, dry_run=True, saver_overwrite=mock_saver)
