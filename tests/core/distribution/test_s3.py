"""Tests for marimba.core.distribution.s3 module."""

from pathlib import Path

import pytest
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError

from marimba.core.distribution.base import DistributionTargetBase
from marimba.core.distribution.s3 import S3DistributionTarget


class TestS3DistributionTarget:
    """Test S3DistributionTarget functionality."""

    @pytest.fixture
    def s3_credentials(self):
        """Provide S3 credentials for testing."""
        return {
            "bucket_name": "test-bucket",
            "endpoint_url": "https://s3.example.com",
            "access_key_id": "test-key",
            "secret_access_key": "test-secret",
            "base_prefix": "datasets",
        }

    @pytest.fixture
    def mock_dataset_wrapper(self, tmp_path, mocker):
        """Create a mock dataset wrapper with test files."""
        dataset_dir = tmp_path / "test_dataset"
        dataset_dir.mkdir()

        # Create test files
        (dataset_dir / "metadata.yaml").write_text("test: data")
        (dataset_dir / "data.txt").write_text("sample data")

        subdir = dataset_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested content")

        mock_wrapper = mocker.Mock()
        mock_wrapper.root_dir = dataset_dir
        mock_wrapper.name = "test_dataset"

        return mock_wrapper

    @pytest.mark.integration
    def test_s3_target_init(self, mocker, s3_credentials):
        """Test S3DistributionTarget initialization."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_bucket = mocker.Mock()
        mock_s3.Bucket.return_value = mock_bucket
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)

        # Verify boto3 resource was created with correct parameters
        mock_resource.assert_called_once_with(
            "s3",
            endpoint_url=s3_credentials["endpoint_url"],
            aws_access_key_id=s3_credentials["access_key_id"],
            aws_secret_access_key=s3_credentials["secret_access_key"],
        )

        # Verify bucket was set up
        mock_s3.Bucket.assert_called_once_with(s3_credentials["bucket_name"])

        # Verify internal state
        assert target._bucket_name == s3_credentials["bucket_name"]
        assert target._base_prefix == s3_credentials["base_prefix"]
        assert target._bucket == mock_bucket

    @pytest.mark.integration
    def test_s3_target_init_strip_prefix(self, mocker, s3_credentials):
        """Test S3DistributionTarget strips trailing slashes from prefix."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        s3_credentials["base_prefix"] = "datasets///"

        target = S3DistributionTarget(**s3_credentials)

        assert target._base_prefix == "datasets"

    @pytest.mark.integration
    def test_check_bucket_success(self, mocker, s3_credentials):
        """Test successful bucket check."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_client = mocker.Mock()
        mock_s3.meta.client = mock_client
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)

        # Should not raise any exception
        target._check_bucket()

        mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")

    @pytest.mark.integration
    def test_check_bucket_error(self, mocker, s3_credentials):
        """Test bucket check with ClientError."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_client = mocker.Mock()
        mock_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
        )
        mock_s3.meta.client = mock_client
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)

        with pytest.raises(ClientError):
            target._check_bucket()

    @pytest.mark.integration
    def test_iterate_dataset_wrapper(self, mocker, s3_credentials, mock_dataset_wrapper):
        """Test iterating over dataset files."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)

        path_key_pairs = list(target._iterate_dataset_wrapper(mock_dataset_wrapper))

        # Should find all files (not directories)
        assert len(path_key_pairs) == 3

        # Check path-key pairs
        paths_and_keys = {str(path.name): key for path, key in path_key_pairs}

        assert "metadata.yaml" in paths_and_keys
        assert "data.txt" in paths_and_keys
        assert "nested.txt" in paths_and_keys

        # Check key structure includes base prefix
        assert paths_and_keys["metadata.yaml"] == "datasets/metadata.yaml"
        assert paths_and_keys["data.txt"] == "datasets/data.txt"
        assert paths_and_keys["nested.txt"] == "datasets/subdir/nested.txt"

    @pytest.mark.integration
    def test_iterate_dataset_wrapper_no_prefix(self, mocker, s3_credentials, mock_dataset_wrapper):
        """Test iterating over dataset files without prefix."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        s3_credentials["base_prefix"] = ""

        mock_s3 = mocker.Mock()
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)

        path_key_pairs = list(target._iterate_dataset_wrapper(mock_dataset_wrapper))

        # Check key structure without base prefix (note: empty prefix creates leading slash)
        paths_and_keys = {str(path.name): key for path, key in path_key_pairs}

        assert paths_and_keys["metadata.yaml"] == "/metadata.yaml"  # Empty prefix creates leading slash
        assert paths_and_keys["data.txt"] == "/data.txt"
        assert paths_and_keys["nested.txt"] == "/subdir/nested.txt"

    @pytest.mark.integration
    def test_upload_success(self, mocker, s3_credentials, tmp_path):
        """Test successful file upload."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_bucket = mocker.Mock()
        mock_s3.Bucket.return_value = mock_bucket
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        target._upload(test_file, "test-key")

        mock_bucket.upload_file.assert_called_once_with(str(test_file.absolute()), "test-key", Config=target._config)

    @pytest.mark.integration
    def test_distribute_success(self, mocker, s3_credentials, mock_dataset_wrapper):
        """Test successful dataset distribution."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_bucket = mocker.Mock()
        mock_s3.Bucket.return_value = mock_bucket
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)
        mock_upload = mocker.patch.object(target, "_upload")
        target.distribute(mock_dataset_wrapper)

        # Should call upload for each file
        assert mock_upload.call_count == 3

        # Check upload calls
        upload_calls = mock_upload.call_args_list
        uploaded_keys = [call[0][1] for call in upload_calls]  # Get the key arguments

        assert "datasets/metadata.yaml" in uploaded_keys
        assert "datasets/data.txt" in uploaded_keys
        assert "datasets/subdir/nested.txt" in uploaded_keys

    @pytest.mark.integration
    def test_distribute_s3_upload_error(self, mocker, s3_credentials, mock_dataset_wrapper):
        """Test distribution with S3 upload error."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_bucket = mocker.Mock()
        mock_s3.Bucket.return_value = mock_bucket
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)
        mock_upload = mocker.patch.object(target, "_upload")
        mock_upload.side_effect = S3UploadFailedError("Upload failed")

        with pytest.raises(DistributionTargetBase.DistributionError, match="S3 upload failed"):
            target.distribute(mock_dataset_wrapper)

    @pytest.mark.integration
    def test_distribute_client_error(self, mocker, s3_credentials, mock_dataset_wrapper):
        """Test distribution with AWS client error."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_bucket = mocker.Mock()
        mock_s3.Bucket.return_value = mock_bucket
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)
        mock_upload = mocker.patch.object(target, "_upload")
        mock_upload.side_effect = ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "PutObject")

        with pytest.raises(DistributionTargetBase.DistributionError, match="AWS client error"):
            target.distribute(mock_dataset_wrapper)

    @pytest.mark.integration
    def test_distribute_generic_error(self, mocker, s3_credentials, mock_dataset_wrapper):
        """Test distribution with generic error."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_bucket = mocker.Mock()
        mock_s3.Bucket.return_value = mock_bucket
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)
        mock_upload = mocker.patch.object(target, "_upload")
        mock_upload.side_effect = Exception("Generic error")

        with pytest.raises(DistributionTargetBase.DistributionError, match="Failed to upload"):
            target.distribute(mock_dataset_wrapper)

    @pytest.mark.integration
    def test_distribute_outer_exception(self, mocker, s3_credentials, mock_dataset_wrapper):
        """Test distribution with exception in outer distribute method."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_bucket = mocker.Mock()
        mock_s3.Bucket.return_value = mock_bucket
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)
        mock_inner = mocker.patch.object(target, "_distribute")
        mock_inner.side_effect = Exception("Inner error")

        with pytest.raises(DistributionTargetBase.DistributionError, match="Distribution error"):
            target.distribute(mock_dataset_wrapper)

    @pytest.mark.integration
    def test_transfer_config_setup(self, mocker, s3_credentials):
        """Test TransferConfig is set up correctly."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)

        assert target._config.multipart_threshold == 100 * 1024 * 1024

    @pytest.mark.integration
    def test_empty_dataset(self, mocker, s3_credentials, tmp_path):
        """Test distribution with empty dataset."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_bucket = mocker.Mock()
        mock_s3.Bucket.return_value = mock_bucket
        mock_resource.return_value = mock_s3

        # Create empty dataset
        empty_dataset = tmp_path / "empty_dataset"
        empty_dataset.mkdir()

        mock_wrapper = mocker.Mock()
        mock_wrapper.root_dir = empty_dataset
        mock_wrapper.name = "empty_dataset"

        target = S3DistributionTarget(**s3_credentials)
        mock_upload = mocker.patch.object(target, "_upload")
        # Should complete without error, no uploads
        target.distribute(mock_wrapper)
        mock_upload.assert_not_called()

    @pytest.mark.integration
    def test_large_file_handling(self, mocker, s3_credentials, mock_dataset_wrapper):
        """Test handling of files larger than multipart threshold."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_bucket = mocker.Mock()
        mock_s3.Bucket.return_value = mock_bucket
        mock_resource.return_value = mock_s3

        target = S3DistributionTarget(**s3_credentials)

        # Mock path.stat() to return large file size with proper st_mode
        mock_stat = mocker.patch("pathlib.Path.stat")
        mock_stat_result = mocker.Mock()
        mock_stat_result.st_size = 200 * 1024 * 1024  # 200MB
        mock_stat_result.st_mode = 0o100644  # Regular file mode
        mock_stat.return_value = mock_stat_result

        # Also need to mock is_dir for the path checking in glob
        mock_is_dir = mocker.patch("pathlib.Path.is_dir", return_value=True)
        mock_upload = mocker.patch.object(target, "_upload")
        target.distribute(mock_dataset_wrapper)

        # Should upload files found in the dataset
        assert mock_upload.call_count >= 1  # At least one file uploaded

    @pytest.mark.integration
    def test_progress_tracking(self, mocker, s3_credentials, mock_dataset_wrapper):
        """Test that progress tracking is set up correctly."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_progress = mocker.patch("marimba.core.distribution.s3.Progress")
        mock_s3 = mocker.Mock()
        mock_bucket = mocker.Mock()
        mock_s3.Bucket.return_value = mock_bucket
        mock_resource.return_value = mock_s3

        # Set up progress mocks
        mock_progress_instance = mocker.Mock()
        mock_progress.return_value.__enter__ = mocker.Mock(return_value=mock_progress_instance)
        mock_progress.return_value.__exit__ = mocker.Mock(return_value=None)

        target = S3DistributionTarget(**s3_credentials)
        mock_upload = mocker.patch.object(target, "_upload")
        target.distribute(mock_dataset_wrapper)

        # Progress should be created and used for collection, size calc, and upload
        assert mock_progress.call_count >= 3
        assert mock_progress_instance.add_task.call_count >= 3
