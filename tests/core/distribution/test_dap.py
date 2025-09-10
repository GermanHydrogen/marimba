"""Tests for marimba.core.distribution.dap module."""

from typing import Any

import pytest
import pytest_mock

from marimba.core.distribution.dap import CSIRODapDistributionTarget
from marimba.core.distribution.s3 import S3DistributionTarget


@pytest.mark.integration
class TestCSIRODapDistributionTarget:
    """Test CSIRODapDistributionTarget functionality."""

    @pytest.fixture
    def dap_credentials(self) -> dict[str, str]:
        """Provide DAP credentials for testing."""
        return {
            "endpoint_url": "https://dap.example.com",
            "access_key": "test_key",
            "secret_access_key": "test_secret",
        }

    @pytest.fixture
    def mock_s3_resource(self, mocker: pytest_mock.MockerFixture) -> tuple[Any, Any, Any]:
        """Mock the boto3 S3 resource creation."""
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_bucket = mocker.Mock()
        mock_s3.Bucket.return_value = mock_bucket
        mock_resource.return_value = mock_s3
        return mock_resource, mock_s3, mock_bucket

    def test_init_with_basic_path(
        self,
        mocker: pytest_mock.MockerFixture,
        dap_credentials: dict[str, str],
        mock_s3_resource: tuple[Any, Any, Any],
    ) -> None:
        """Test initialization with a basic remote directory path."""
        # Arrange
        mock_resource, mock_s3, mock_bucket = mock_s3_resource
        remote_directory = "bucket_name/path/to/data"

        # Act
        target = CSIRODapDistributionTarget(
            remote_directory=remote_directory,
            **dap_credentials,
        )

        # Assert
        assert isinstance(target, S3DistributionTarget), "Should inherit from S3DistributionTarget"
        assert isinstance(target, CSIRODapDistributionTarget), "Should be instance of CSIRODapDistributionTarget"
        assert target._bucket_name == "bucket_name", "Bucket name should be parsed correctly"
        assert target._base_prefix == "path/to/data", "Base prefix should be parsed correctly"

        # Verify S3 resource was created with correct parameters
        mock_resource.assert_called_once_with(
            "s3",
            endpoint_url=dap_credentials["endpoint_url"],
            aws_access_key_id=dap_credentials["access_key"],
            aws_secret_access_key=dap_credentials["secret_access_key"],
        )

    def test_init_with_root_path(
        self,
        mocker: pytest_mock.MockerFixture,
        dap_credentials: dict[str, str],
        mock_s3_resource: tuple[Any, Any, Any],
    ) -> None:
        """Test initialization with root-level remote directory."""
        # Arrange
        mock_resource, mock_s3, mock_bucket = mock_s3_resource
        remote_directory = "bucket_name/"

        # Act
        target = CSIRODapDistributionTarget(
            remote_directory=remote_directory,
            **dap_credentials,
        )

        # Assert
        assert target._bucket_name == "bucket_name", "Bucket name should be parsed correctly"
        assert target._base_prefix == "", "Base prefix should be empty for root path"

    def test_init_with_nested_path(
        self,
        mocker: pytest_mock.MockerFixture,
        dap_credentials: dict[str, str],
        mock_s3_resource: tuple[Any, Any, Any],
    ) -> None:
        """Test initialization with deeply nested remote directory."""
        # Arrange
        mock_resource, mock_s3, mock_bucket = mock_s3_resource
        remote_directory = "my-bucket/data/2023/project-x/results"

        # Act
        target = CSIRODapDistributionTarget(
            remote_directory=remote_directory,
            **dap_credentials,
        )

        # Assert
        assert target._bucket_name == "my-bucket", "Bucket name should be parsed correctly"
        assert target._base_prefix == "data/2023/project-x/results", "Base prefix should handle nested paths"

    def test_init_with_bucket_only_no_slash(
        self,
        mocker: pytest_mock.MockerFixture,
        dap_credentials: dict[str, str],
        mock_s3_resource: tuple[Any, Any, Any],
    ) -> None:
        """Test initialization with bucket name only (no slash) - documents current implementation behavior.

        NOTE: This test documents the current behavior where bucket names without slashes
        get truncated. This may be unintended behavior that should be fixed.
        """
        # Arrange
        mock_resource, mock_s3, mock_bucket = mock_s3_resource
        remote_directory = "bucket_name"

        # Act
        target = CSIRODapDistributionTarget(
            remote_directory=remote_directory,
            **dap_credentials,
        )

        # Assert
        # When find("/") returns -1, slicing [:first_slash] becomes [:-1] which removes last character
        assert target._bucket_name == "bucket_nam", "Bucket name gets last character removed when no slash found"
        assert target._base_prefix == "bucket_name", "Base prefix becomes the full string when no slash found"

    def test_remote_directory_parsing_edge_cases(
        self,
        mocker: pytest_mock.MockerFixture,
        dap_credentials: dict[str, str],
        mock_s3_resource: tuple[Any, Any, Any],
    ) -> None:
        """Test edge cases in remote directory parsing."""
        # Arrange
        mock_resource, mock_s3, mock_bucket = mock_s3_resource
        multiple_slash_path = "bucket/path/with/multiple/slashes"
        leading_slash_path = "/bucket/path"

        # Act
        target1 = CSIRODapDistributionTarget(
            endpoint_url=dap_credentials["endpoint_url"],
            access_key=dap_credentials["access_key"],
            secret_access_key=dap_credentials["secret_access_key"],
            remote_directory=multiple_slash_path,
        )

        target2 = CSIRODapDistributionTarget(
            endpoint_url=dap_credentials["endpoint_url"],
            access_key=dap_credentials["access_key"],
            secret_access_key=dap_credentials["secret_access_key"],
            remote_directory=leading_slash_path,
        )

        # Assert
        # Test with multiple slashes
        assert target1._bucket_name == "bucket", "Should use first slash for bucket separation"
        assert target1._base_prefix == "path/with/multiple/slashes", "Should keep remaining path as prefix"

        # Test with slash at beginning - edge case
        assert target2._bucket_name == "", "Bucket name becomes empty when path starts with slash"
        assert target2._base_prefix == "bucket/path", "Base prefix contains everything after first slash"

    def test_all_parameters_passed_to_parent(self, mocker: pytest_mock.MockerFixture) -> None:
        """Test that all parameters are correctly passed to parent S3DistributionTarget."""
        # Arrange
        mock_resource = mocker.patch("marimba.core.distribution.s3.resource")
        mock_s3 = mocker.Mock()
        mock_bucket = mocker.Mock()
        mock_s3.Bucket.return_value = mock_bucket
        mock_resource.return_value = mock_s3

        endpoint = "https://custom.dap.server.com"
        access_key = "custom_access_key"
        secret_key = "custom_secret_key"
        remote_dir = "test-bucket/data/experiment"

        # Act
        target = CSIRODapDistributionTarget(
            endpoint_url=endpoint,
            access_key=access_key,
            secret_access_key=secret_key,
            remote_directory=remote_dir,
        )

        # Assert
        # Verify parameters were processed correctly
        assert target._bucket_name == "test-bucket", "Bucket name should be parsed from remote directory"
        assert target._base_prefix == "data/experiment", "Base prefix should be parsed from remote directory"

        # Verify parent constructor was called with correct parameters
        mock_resource.assert_called_once_with(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,  # Note: DAP uses access_key but S3 expects access_key_id
            aws_secret_access_key=secret_key,
        )

        # Verify S3 resource and bucket were created
        assert hasattr(target, "_s3"), "Should have _s3 attribute from parent"
        assert hasattr(target, "_bucket"), "Should have _bucket attribute from parent"
        assert target._s3 is mock_s3, "Should store the mocked S3 resource"
        assert target._bucket is mock_bucket, "Should store the mocked S3 bucket"

    def test_inherits_s3_distribution_methods(
        self,
        mocker: pytest_mock.MockerFixture,
        dap_credentials: dict[str, str],
        mock_s3_resource: tuple[Any, Any, Any],
    ) -> None:
        """Test that CSIRODapDistributionTarget inherits methods from S3DistributionTarget."""
        # Arrange
        remote_directory = "bucket/path"

        # Act
        target = CSIRODapDistributionTarget(
            remote_directory=remote_directory,
            **dap_credentials,
        )

        # Assert
        # Check that all expected S3DistributionTarget methods are available
        assert hasattr(target, "distribute"), "Should inherit distribute method from S3DistributionTarget"
        assert callable(target.distribute), "distribute should be callable"

        # Verify internal attributes set by parent constructor
        assert hasattr(target, "_bucket"), "Should have _bucket attribute from parent"
        assert hasattr(target, "_s3"), "Should have _s3 attribute from parent"
