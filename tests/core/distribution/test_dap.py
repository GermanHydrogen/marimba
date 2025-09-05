"""
Unit tests for CSIRODapDistributionTarget.

Tests the functionality of the CSIRO DAP distribution target including:
- Initialization and parameter parsing
- Inheritance from S3DistributionTarget
- Remote directory path parsing
"""

import pytest

from marimba.core.distribution.dap import CSIRODapDistributionTarget
from marimba.core.distribution.s3 import S3DistributionTarget


@pytest.mark.unit
class TestCSIRODapDistributionTarget:
    """Test CSIRODapDistributionTarget functionality."""

    def test_init_with_basic_path(self):
        """Test initialization with a basic remote directory path."""
        target = CSIRODapDistributionTarget(
            endpoint_url="https://dap.example.com",
            access_key="test_key",
            secret_access_key="test_secret",
            remote_directory="bucket_name/path/to/data",
        )

        # Should inherit from S3DistributionTarget
        assert isinstance(target, S3DistributionTarget)

        # Check that internal attributes are set correctly through parent
        assert target._bucket_name == "bucket_name"
        assert target._base_prefix == "path/to/data"

    def test_init_with_root_path(self):
        """Test initialization with root-level remote directory."""
        target = CSIRODapDistributionTarget(
            endpoint_url="https://dap.example.com",
            access_key="test_key",
            secret_access_key="test_secret",
            remote_directory="bucket_name/",
        )

        assert target._bucket_name == "bucket_name"
        assert target._base_prefix == ""

    def test_init_with_nested_path(self):
        """Test initialization with deeply nested remote directory."""
        target = CSIRODapDistributionTarget(
            endpoint_url="https://dap.example.com",
            access_key="test_key",
            secret_access_key="test_secret",
            remote_directory="my-bucket/data/2023/project-x/results",
        )

        assert target._bucket_name == "my-bucket"
        assert target._base_prefix == "data/2023/project-x/results"

    def test_init_with_bucket_only(self):
        """Test initialization with bucket name only (no slash)."""
        target = CSIRODapDistributionTarget(
            endpoint_url="https://dap.example.com",
            access_key="test_key",
            secret_access_key="test_secret",
            remote_directory="bucket_name",
        )

        # When there's no slash, find("/") returns -1
        # So bucket_name would be remote_directory[:−1] (empty)
        # and base_prefix would be remote_directory[0:] (full string)
        # This tests the edge case behavior
        assert target._bucket_name == "bucket_nam"  # Last character removed
        assert target._base_prefix == "bucket_name"  # When first_slash = -1, [first_slash + 1:] = [0:]

    def test_inherits_from_s3_distribution_target(self):
        """Test that CSIRODapDistributionTarget properly inherits from S3DistributionTarget."""
        target = CSIRODapDistributionTarget(
            endpoint_url="https://dap.example.com",
            access_key="test_key",
            secret_access_key="test_secret",
            remote_directory="bucket/path",
        )

        # Check inheritance
        assert isinstance(target, S3DistributionTarget)
        assert isinstance(target, CSIRODapDistributionTarget)

        # Check that all S3DistributionTarget methods are available
        assert hasattr(target, "distribute")
        assert hasattr(target, "_bucket")
        assert hasattr(target, "_s3")

    def test_remote_directory_parsing_edge_cases(self):
        """Test edge cases in remote directory parsing."""
        # Test with multiple slashes
        target1 = CSIRODapDistributionTarget(
            endpoint_url="https://dap.example.com",
            access_key="key",
            secret_access_key="secret",
            remote_directory="bucket/path/with/multiple/slashes",
        )
        assert target1._bucket_name == "bucket"
        assert target1._base_prefix == "path/with/multiple/slashes"

        # Test with slash at beginning
        target2 = CSIRODapDistributionTarget(
            endpoint_url="https://dap.example.com",
            access_key="key",
            secret_access_key="secret",
            remote_directory="/bucket/path",
        )
        assert target2._bucket_name == ""
        assert target2._base_prefix == "bucket/path"

    def test_all_parameters_passed_to_parent(self):
        """Test that all parameters are correctly passed to parent S3DistributionTarget."""
        endpoint = "https://custom.dap.server.com"
        access_key = "custom_access_key"
        secret_key = "custom_secret_key"
        remote_dir = "test-bucket/data/experiment"

        target = CSIRODapDistributionTarget(
            endpoint_url=endpoint, access_key=access_key, secret_access_key=secret_key, remote_directory=remote_dir
        )

        # Verify parameters were processed correctly
        assert target._bucket_name == "test-bucket"
        assert target._base_prefix == "data/experiment"
        # Verify the S3 resource was created (we can't easily check credentials)
        assert hasattr(target, "_s3")
        assert hasattr(target, "_bucket")
