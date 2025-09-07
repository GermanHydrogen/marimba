"""Tests for marimba.core.schemas.base module."""

from datetime import datetime

import pytest

from marimba.core.schemas.base import BaseMetadata


class TestBaseMetadata:
    """Test BaseMetadata abstract base class."""

    @pytest.mark.unit
    def test_base_metadata_is_abstract(self):
        """Test that BaseMetadata cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetadata()  # type: ignore[abstract]  # Should raise TypeError for abstract class

    @pytest.mark.unit
    def test_concrete_implementation_required_properties(self):
        """Test that concrete implementations must implement all abstract properties."""

        class IncompleteMetadata(BaseMetadata):
            """Incomplete implementation missing some properties."""

            @property
            def datetime(self):
                return datetime.now()

            # Missing other required properties

        with pytest.raises(TypeError):
            IncompleteMetadata()  # type: ignore[abstract]  # Should fail due to missing abstract methods

    @pytest.mark.unit
    def test_complete_implementation(self):
        """Test that a complete implementation can be instantiated."""

        class CompleteMetadata(BaseMetadata):
            """Complete implementation of BaseMetadata."""

            def __init__(self):
                self._hash = None

            @property
            def datetime(self):
                return datetime(2023, 1, 1, 12, 0, 0)

            @property
            def latitude(self):
                return 37.7749

            @property
            def longitude(self):
                return -122.4194

            @property
            def altitude(self):
                return 100.0

            @property
            def context(self):
                return "Test context"

            @property
            def license(self):
                return "MIT"

            @property
            def creators(self):
                return ["Test Creator"]

            @property
            def hash_sha256(self):
                return self._hash

            @hash_sha256.setter
            def hash_sha256(self, value):
                self._hash = value

            @classmethod
            def create_dataset_metadata(
                cls,
                dataset_name,
                root_dir,
                items,
                metadata_name=None,
                *,
                dry_run=False,
                saver_overwrite=None,
            ):
                pass

            @classmethod
            def process_files(cls, dataset_mapping, max_workers=None, logger=None, *, dry_run=False, chunk_size=None):
                pass

        # Should be able to instantiate
        metadata = CompleteMetadata()  # type: ignore[no-untyped-call]

        # Test property access
        assert metadata.datetime == datetime(2023, 1, 1, 12, 0, 0)
        assert metadata.latitude == 37.7749
        assert metadata.longitude == -122.4194
        assert metadata.altitude == 100.0
        assert metadata.context == "Test context"
        assert metadata.license == "MIT"
        assert metadata.creators == ["Test Creator"]

        # Test hash property with setter
        assert metadata.hash_sha256 is None
        metadata.hash_sha256 = "test_hash"
        assert metadata.hash_sha256 == "test_hash"

    @pytest.mark.unit
    def test_abstract_property_requirements(self):
        """Test that all expected abstract properties are defined."""
        expected_properties = {
            "datetime",
            "latitude",
            "longitude",
            "altitude",
            "context",
            "license",
            "creators",
            "hash_sha256",
        }

        # Get abstract properties from the class
        abstract_properties = set()
        for name in dir(BaseMetadata):
            attr = getattr(BaseMetadata, name)
            if hasattr(attr, "__isabstractmethod__") and attr.__isabstractmethod__:
                abstract_properties.add(name)

        # Check that all expected properties are abstract
        for prop in expected_properties:
            assert prop in abstract_properties, f"Property {prop} should be abstract"

    @pytest.mark.unit
    def test_abstract_method_requirements(self):
        """Test that all expected abstract methods are defined."""
        expected_methods = {"create_dataset_metadata", "process_files"}

        # Get abstract methods from the class
        abstract_methods = set()
        for name in dir(BaseMetadata):
            attr = getattr(BaseMetadata, name)
            if hasattr(attr, "__isabstractmethod__") and attr.__isabstractmethod__:
                abstract_methods.add(name)

        # Check that all expected methods are abstract
        for method in expected_methods:
            assert method in abstract_methods, f"Method {method} should be abstract"

    @pytest.mark.unit
    def test_create_dataset_metadata_signature(self):
        """Test the signature of create_dataset_metadata abstract method."""
        # This tests that the method signature is as expected
        import inspect

        method = BaseMetadata.create_dataset_metadata
        sig = inspect.signature(method)

        # Check parameter names (cls is implicit in classmethod)
        param_names = list(sig.parameters.keys())
        expected_params = ["dataset_name", "root_dir", "items", "metadata_name"]

        # Check that all expected parameters are present
        for param in expected_params:
            assert param in param_names, f"Parameter {param} should be in signature"

        # Check keyword-only parameters
        params = sig.parameters
        assert params["dry_run"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["saver_overwrite"].kind == inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.unit
    def test_process_files_signature(self):
        """Test the signature of process_files abstract method."""
        import inspect

        method = BaseMetadata.process_files
        sig = inspect.signature(method)

        # Check parameter names (cls is implicit in classmethod)
        param_names = list(sig.parameters.keys())
        expected_params = ["dataset_mapping"]

        # Check that required parameters are present
        for param in expected_params:
            assert param in param_names, f"Parameter {param} should be in signature"

        # Check keyword-only parameters
        params = sig.parameters
        assert params["dry_run"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["chunk_size"].kind == inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.unit
    def test_base_metadata_inheritance_structure(self):
        """Test that BaseMetadata properly inherits from ABC."""
        from abc import ABC

        assert issubclass(BaseMetadata, ABC)
        assert hasattr(BaseMetadata, "__abstractmethods__")
        assert len(BaseMetadata.__abstractmethods__) > 0

    @pytest.mark.unit
    def test_property_return_types_validation(self):
        """Test validation of property return types in implementation."""

        class TypedMetadata(BaseMetadata):
            """Implementation with specific return types for testing."""

            @property
            def datetime(self):
                return None  # Should allow None

            @property
            def latitude(self):
                return None  # Should allow None

            @property
            def longitude(self):
                return None  # Should allow None

            @property
            def altitude(self):
                return None  # Should allow None

            @property
            def context(self):
                return None  # Should allow None

            @property
            def license(self):
                return None  # Should allow None

            @property
            def creators(self):
                return []  # Should return list

            @property
            def hash_sha256(self):
                return None  # Should allow None

            @hash_sha256.setter
            def hash_sha256(self, value):
                pass

            @classmethod
            def create_dataset_metadata(
                cls,
                dataset_name,
                root_dir,
                items,
                metadata_name=None,
                *,
                dry_run=False,
                saver_overwrite=None,
            ):
                pass

            @classmethod
            def process_files(cls, dataset_mapping, max_workers=None, logger=None, *, dry_run=False, chunk_size=None):
                pass

        metadata = TypedMetadata()

        # Test that None values are acceptable
        assert metadata.datetime is None
        assert metadata.latitude is None
        assert metadata.longitude is None
        assert metadata.altitude is None
        assert metadata.context is None
        assert metadata.license is None
        assert metadata.hash_sha256 is None

        # Test that creators returns a list
        assert isinstance(metadata.creators, list)
        assert metadata.creators == []
