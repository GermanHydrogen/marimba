"""Tests for marimba.core.schemas.generic module."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from marimba.core.schemas.base import BaseMetadata
from marimba.core.schemas.generic import GenericMetadata


class TestGenericMetadata:
    """Test GenericMetadata class."""

    @pytest.fixture
    def sample_datetime(self):
        """Sample datetime for testing."""
        return datetime(2024, 1, 15, 12, 30, 45)

    @pytest.fixture
    def basic_metadata(self, sample_datetime):
        """Basic metadata instance for testing."""
        return GenericMetadata(
            datetime_=sample_datetime,
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0,
            context="Test context",
            license_="MIT",
            creators=["Alice", "Bob"],
            hash_sha256_="abc123",
        )

    @pytest.mark.unit
    def test_initialization_with_all_parameters(self, sample_datetime):
        """Test initialization with all parameters."""
        metadata = GenericMetadata(
            datetime_=sample_datetime,
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0,
            context="Test context",
            license_="MIT",
            creators=["Alice", "Bob"],
            hash_sha256_="abc123",
        )

        assert metadata.datetime == sample_datetime
        assert metadata.latitude == 37.7749
        assert metadata.longitude == -122.4194
        assert metadata.altitude == 100.0
        assert metadata.context == "Test context"
        assert metadata.license == "MIT"
        assert metadata.creators == ["Alice", "Bob"]

    @pytest.mark.unit
    def test_initialization_with_minimal_parameters(self):
        """Test initialization with minimal parameters."""
        metadata = GenericMetadata()

        assert metadata.datetime is None
        assert metadata.latitude is None
        assert metadata.longitude is None
        assert metadata.altitude is None
        assert metadata.context is None
        assert metadata.license is None
        assert metadata.creators == []
        assert metadata.hash_sha256 is None

    @pytest.mark.unit
    def test_hash_string_conversion_hex(self):
        """Test hash string conversion from hex."""
        metadata = GenericMetadata(hash_sha256_="abc123")
        # The hex conversion should work
        assert metadata is not None

    @pytest.mark.unit
    def test_hash_string_conversion_invalid_hex(self):
        """Test hash string conversion from invalid hex (falls back to utf-8)."""
        metadata = GenericMetadata(hash_sha256_="invalid_hex_string")
        # Should still create metadata instance without error
        assert metadata is not None

    @pytest.mark.unit
    def test_hash_bytes_input(self):
        """Test hash input as bytes."""
        hash_bytes = b"test_hash"
        metadata = GenericMetadata(hash_sha256_=hash_bytes)
        assert metadata is not None

    @pytest.mark.unit
    def test_strftime_with_datetime(self, basic_metadata):
        """Test strftime formatting with datetime."""
        result = basic_metadata.strftime("%Y-%m-%d")
        assert result == "2024-01-15"

    @pytest.mark.unit
    def test_strftime_without_datetime(self):
        """Test strftime formatting without datetime raises ValueError."""
        metadata = GenericMetadata()
        with pytest.raises(ValueError, match="Cannot format datetime: datetime is None"):
            metadata.strftime("%Y-%m-%d")

    @pytest.mark.unit
    def test_isoformat_with_datetime(self, basic_metadata):
        """Test isoformat with datetime."""
        result = basic_metadata.isoformat()
        assert result == "2024-01-15T12:30:45"

    @pytest.mark.unit
    def test_isoformat_without_datetime(self):
        """Test isoformat without datetime raises ValueError."""
        metadata = GenericMetadata()
        with pytest.raises(ValueError, match="Cannot format datetime: datetime is None"):
            metadata.isoformat()

    @pytest.mark.unit
    def test_comparison_operators_with_datetime(self, sample_datetime):
        """Test comparison operators with datetime objects."""
        earlier_dt = datetime(2024, 1, 14, 12, 30, 45)
        later_dt = datetime(2024, 1, 16, 12, 30, 45)

        metadata = GenericMetadata(datetime_=sample_datetime)

        # Test with datetime objects
        assert metadata > earlier_dt
        assert metadata < later_dt
        assert metadata == sample_datetime
        assert metadata <= sample_datetime
        assert metadata >= sample_datetime

    @pytest.mark.unit
    def test_comparison_operators_with_metadata(self, sample_datetime):
        """Test comparison operators with other GenericMetadata objects."""
        earlier_dt = datetime(2024, 1, 14, 12, 30, 45)
        later_dt = datetime(2024, 1, 16, 12, 30, 45)

        metadata = GenericMetadata(datetime_=sample_datetime)
        earlier_metadata = GenericMetadata(datetime_=earlier_dt)
        later_metadata = GenericMetadata(datetime_=later_dt)
        same_metadata = GenericMetadata(datetime_=sample_datetime)

        assert metadata > earlier_metadata
        assert metadata < later_metadata
        assert metadata == same_metadata
        assert metadata <= same_metadata
        assert metadata >= same_metadata

    @pytest.mark.unit
    def test_comparison_operators_with_none_datetime(self):
        """Test comparison operators when datetime is None."""
        metadata_none = GenericMetadata()
        metadata_with_dt = GenericMetadata(datetime_=datetime(2024, 1, 15))
        other_metadata_none = GenericMetadata()

        # None datetime should be less than any datetime
        assert metadata_none < metadata_with_dt
        assert not (metadata_none > metadata_with_dt)
        assert not (metadata_with_dt < metadata_none)
        assert metadata_with_dt > metadata_none

    @pytest.mark.unit
    def test_comparison_operators_both_none_datetime(self):
        """Test comparison operators when both have None datetime."""
        metadata1 = GenericMetadata()
        metadata2 = GenericMetadata()

        # Both None should be equal
        assert metadata1 == metadata2
        assert metadata1 <= metadata2
        assert metadata1 >= metadata2

    @pytest.mark.unit
    def test_comparison_operators_invalid_type(self, basic_metadata):
        """Test comparison operators with invalid types."""
        assert basic_metadata.__lt__("invalid") == NotImplemented
        assert basic_metadata.__gt__("invalid") == NotImplemented
        assert basic_metadata.__eq__("invalid") == NotImplemented

    @pytest.mark.unit
    def test_hash_method(self, sample_datetime):
        """Test __hash__ method enables use in sets and as dict keys."""
        metadata1 = GenericMetadata(datetime_=sample_datetime)
        metadata2 = GenericMetadata(datetime_=sample_datetime)

        # Same datetime should have same hash
        assert hash(metadata1) == hash(metadata2)

        # Should be usable in sets
        metadata_set = {metadata1, metadata2}
        assert len(metadata_set) == 1

    @pytest.mark.unit
    def test_hash_sha256_setter(self):
        """Test hash_sha256 setter."""
        metadata = GenericMetadata()
        metadata.hash_sha256 = "new_hash_value"
        assert metadata.hash_sha256 == "new_hash_value"

    @pytest.mark.unit
    def test_format_hash_with_value(self, basic_metadata):
        """Test format_hash with hash value."""
        # This should return the hash value as string
        result = basic_metadata.format_hash()
        assert result is not None

    @pytest.mark.unit
    def test_format_hash_without_value(self):
        """Test format_hash without hash value."""
        metadata = GenericMetadata()
        result = metadata.format_hash()
        assert result is None

    @pytest.mark.unit
    def test_create_dataset_metadata_dry_run(self, tmp_path, sample_datetime):
        """Test create_dataset_metadata with dry_run=True."""
        items: dict[str, list[BaseMetadata]] = {
            "file1.jpg": [GenericMetadata(datetime_=sample_datetime, latitude=37.7749)],
        }

        # Should not raise error and not create files
        GenericMetadata.create_dataset_metadata(
            dataset_name="test_dataset",
            root_dir=tmp_path,
            items=items,
            dry_run=True,
        )

        # No metadata file should be created in dry run
        metadata_files = list(tmp_path.glob("metadata*"))
        assert len(metadata_files) == 0

    @pytest.mark.unit
    def test_create_dataset_metadata_with_custom_saver(self, mocker, tmp_path, sample_datetime):
        """Test create_dataset_metadata with custom saver."""
        mock_saver = mocker.Mock()
        items: dict[str, list[BaseMetadata]] = {
            "file1.jpg": [GenericMetadata(datetime_=sample_datetime, latitude=37.7749)],
        }

        GenericMetadata.create_dataset_metadata(
            dataset_name="test_dataset",
            root_dir=tmp_path,
            items=items,
            saver_overwrite=mock_saver,
        )

        # Custom saver should be called
        mock_saver.assert_called_once()

    @pytest.mark.unit
    def test_create_dataset_metadata_with_custom_name(self, mocker, tmp_path, sample_datetime):
        """Test create_dataset_metadata with custom metadata name."""
        items: dict[str, list[BaseMetadata]] = {"file1.jpg": [GenericMetadata(datetime_=sample_datetime)]}

        mock_saver = mocker.patch("marimba.core.schemas.generic.yaml_saver")
        GenericMetadata.create_dataset_metadata(
            dataset_name="test_dataset",
            root_dir=tmp_path,
            items=items,
            metadata_name="custom_metadata",
        )

        # Should use custom name
        mock_saver.assert_called_once()
        call_args = mock_saver.call_args
        assert call_args[0][1] == "custom_metadata"

    @pytest.mark.unit
    def test_create_dataset_metadata_with_items_missing_format_hash(self, mocker, tmp_path):
        """Test create_dataset_metadata with items missing format_hash method."""
        # Mock metadata without format_hash method
        mock_metadata = mocker.Mock(spec=BaseMetadata)
        mock_metadata.datetime = None
        mock_metadata.latitude = None
        mock_metadata.longitude = None
        mock_metadata.altitude = None
        mock_metadata.context = None
        mock_metadata.license = None
        mock_metadata.creators = []

        items: dict[str, list[BaseMetadata]] = {"file1.jpg": [mock_metadata]}

        mock_saver = mocker.patch("marimba.core.schemas.generic.yaml_saver")
        GenericMetadata.create_dataset_metadata(dataset_name="test_dataset", root_dir=tmp_path, items=items)

        # Should handle missing format_hash gracefully
        mock_saver.assert_called_once()

    @pytest.mark.unit
    def test_process_files_method(self):
        """Test process_files method (currently just a pass statement)."""
        dataset_mapping: dict[Path, tuple[list[BaseMetadata], dict[str, Any] | None]] = {}

        # Should not raise error
        GenericMetadata.process_files(
            dataset_mapping=dataset_mapping,
            max_workers=1,
            logger=None,
            dry_run=True,
            chunk_size=10,
        )
