"""
Global test fixtures for the marimba test suite.

This module provides shared fixtures used across all test modules,
including common test data, temporary directories, and testing utilities.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List

import pytest
import pytest_mock
from typer.testing import CliRunner


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_project_config() -> Dict[str, Any]:
    """Sample project configuration for testing."""
    return {
        "name": "test_project",
        "version": "1.0.0",
        "description": "A test project for unit testing",
    }


@pytest.fixture
def sample_ifdo_metadata() -> Dict[str, Any]:
    """Sample iFDO metadata for testing."""
    return {
        "image-set-header": {
            "image-set-name": "test_dataset",
            "image-set-uuid": "550e8400-e29b-41d4-a716-446655440000",
            "image-acquisition": {
                "image-acquisition-uuid": "550e8400-e29b-41d4-a716-446655440001",
                "image-coordinate-system": {"coordinate-system-name": "WGS84"},
                "image-capturing": {
                    "image-capturing-uuid": "550e8400-e29b-41d4-a716-446655440002",
                    "image-camera": {
                        "camera-uuid": "550e8400-e29b-41d4-a716-446655440003",
                        "camera-name": "Test Camera",
                        "camera-model": "TestCam 1000",
                    },
                },
            },
        },
        "image-set-items": [],
    }


@pytest.fixture
def sample_test_data_files(temp_dir: Path) -> Path:
    """Create sample test data files in a temporary directory."""
    data_dir = temp_dir / "test_data"
    data_dir.mkdir(parents=True)

    # Create sample images
    (data_dir / "image001.jpg").write_bytes(b"fake_jpeg_data")
    (data_dir / "image002.jpg").write_bytes(b"fake_jpeg_data_2")
    (data_dir / "image003.png").write_bytes(b"fake_png_data")

    # Create metadata files
    (data_dir / "metadata.csv").write_text("filename,timestamp,depth\nimage001.jpg,2024-01-01T10:00:00Z,10.5\n")
    (data_dir / "config.yml").write_text("site_id: TEST01\nfield_of_view: 1000\n")

    return data_dir


@pytest.fixture
def mock_git_operations(mocker: pytest_mock.MockerFixture) -> Dict[str, Any]:
    """Mock Git operations to avoid network dependencies in tests."""

    def mock_clone_from(url: str, to_path: str, **kwargs: Any) -> Any:
        """Mock git clone that creates expected directory structure."""
        repo_path = Path(to_path)
        repo_path.mkdir(parents=True, exist_ok=True)

        # Create basic pipeline structure
        (repo_path / "pipeline.yml").write_text(
            """
name: test_pipeline
version: 1.0.0
description: Test pipeline for unit testing
requirements:
  - python>=3.8
"""
        )
        (repo_path / "main.py").write_text("# Test pipeline main script")

        return mocker.Mock()

    mock_clone = mocker.patch("git.Repo.clone_from", side_effect=mock_clone_from)
    mock_repo = mocker.patch("git.Repo")

    return {
        "clone": mock_clone,
        "repo": mock_repo,
    }


@pytest.fixture
def cli_runner() -> CliRunner:
    """CLI runner for testing typer commands."""
    return CliRunner()


# Test data constants
TEST_COORDINATES = [
    (151.2093, -33.8688),  # Sydney
    (-74.0060, 40.7128),  # New York
    (2.3522, 48.8566),  # Paris
]

TEST_TIMESTAMPS = [
    "2024-01-01T00:00:00Z",
    "2024-06-15T12:30:45Z",
    "2024-12-31T23:59:59Z",
]

# Test file extensions for various operations
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]


# Test Data Factories
class TestDataFactory:
    """Factory for creating consistent test data across the test suite."""

    @staticmethod
    def create_project_config(**overrides: Any) -> Dict[str, Any]:
        """Create a project configuration with optional overrides."""
        config = {
            "name": "test_project",
            "version": "1.0.0",
            "description": "Test project for unit testing",
            "author": "test_user",
            "created": "2024-01-01T00:00:00Z",
        }
        config.update(overrides)
        return config

    @staticmethod
    def create_pipeline_config(**overrides: Any) -> Dict[str, Any]:
        """Create a pipeline configuration with optional overrides."""
        config = {
            "name": "test_pipeline",
            "version": "1.0.0",
            "description": "Test pipeline for unit testing",
            "requirements": ["python>=3.8"],
            "parameters": {"threshold": 0.5, "max_depth": 100, "site_id": "TEST_SITE_01"},
        }
        config.update(overrides)
        return config

    @staticmethod
    def create_collection_config(**overrides: Any) -> Dict[str, Any]:
        """Create a collection configuration with optional overrides."""
        config = {
            "name": "test_collection",
            "site_id": "TEST_SITE_01",
            "field_of_view": "1000",
            "instrument_type": "camera",
            "operation": "copy",
            "created": "2024-01-01T00:00:00Z",
        }
        config.update(overrides)
        return config

    @staticmethod
    def create_ifdo_metadata(**overrides: Any) -> Dict[str, Any]:
        """Create iFDO metadata structure with optional overrides."""
        metadata = {
            "image-set-header": {
                "image-set-name": "test_dataset",
                "image-set-uuid": "550e8400-e29b-41d4-a716-446655440000",
                "image-acquisition": {
                    "image-acquisition-uuid": "550e8400-e29b-41d4-a716-446655440001",
                    "image-coordinate-system": {"coordinate-system-name": "WGS84"},
                    "image-capturing": {
                        "image-capturing-uuid": "550e8400-e29b-41d4-a716-446655440002",
                        "image-camera": {
                            "camera-uuid": "550e8400-e29b-41d4-a716-446655440003",
                            "camera-name": "Test Camera",
                            "camera-model": "TestCam 1000",
                        },
                    },
                },
            },
            "image-set-items": [],
        }
        # Deep merge overrides
        if overrides:
            TestDataFactory._deep_update(metadata, overrides)
        return metadata

    @staticmethod
    def create_dataset_metadata(**overrides: Any) -> Dict[str, Any]:
        """Create dataset metadata with optional overrides."""
        metadata = {
            "name": "test_dataset",
            "version": "1.0.0",
            "description": "Test dataset for unit testing",
            "contact": {"name": "Test User", "email": "test@example.com"},
            "created": "2024-01-01T00:00:00Z",
            "format": "ifdo",
            "license": "CC-BY-4.0",
        }
        metadata.update(overrides)
        return metadata

    @staticmethod
    def create_test_files(base_dir: Path, file_count: int = 3) -> List[Path]:
        """Create test files in a directory."""
        base_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for i in range(file_count):
            file_path = base_dir / f"test_file_{i:03d}.txt"
            file_path.write_text(f"Test file content {i}")
            files.append(file_path)
        return files

    @staticmethod
    def create_test_images(base_dir: Path, image_count: int = 3) -> List[Path]:
        """Create fake test image files."""
        base_dir.mkdir(parents=True, exist_ok=True)
        images = []
        for i in range(image_count):
            image_path = base_dir / f"image_{i:03d}.jpg"
            # Create fake JPEG data (minimal valid JPEG header)
            jpeg_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xd9"
            image_path.write_bytes(jpeg_data)
            images.append(image_path)
        return images

    @staticmethod
    def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Deep update a dictionary with another dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                TestDataFactory._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


@pytest.fixture
def test_data_factory() -> TestDataFactory:
    """Provide the TestDataFactory for tests."""
    return TestDataFactory()


# Common test helper functions
def assert_project_structure_exists(project_dir: Path, message_prefix: str = "") -> None:
    """Assert that a project has the expected directory structure."""
    prefix = f"{message_prefix}: " if message_prefix else ""
    assert project_dir.exists(), f"{prefix}Project directory should exist"
    assert (project_dir / ".marimba").exists(), f"{prefix}Marimba config directory should exist"
    assert (project_dir / "pipelines").exists(), f"{prefix}Pipelines directory should exist"
    assert (project_dir / "collections").exists(), f"{prefix}Collections directory should exist"
    assert (project_dir / "datasets").exists(), f"{prefix}Datasets directory should exist"
    assert (project_dir / "targets").exists(), f"{prefix}Targets directory should exist"


def assert_collection_exists(project_dir: Path, collection_name: str) -> Path:
    """Assert that a collection exists and return its path."""
    collection_dir = project_dir / "collections" / collection_name
    assert collection_dir.exists(), f"Collection {collection_name} directory should exist"

    collection_config = collection_dir / "collection.yml"
    assert collection_config.exists(), f"Collection {collection_name} config should exist"

    return collection_dir


def create_test_project_with_cli(runner: CliRunner, project_dir: Path) -> None:
    """Create a test project using the CLI and verify it was created correctly."""
    from marimba.main import marimba_cli as app

    result = runner.invoke(app, ["new", "project", str(project_dir)])
    assert result.exit_code == 0, f"Project creation should succeed: {result.stdout}"
    assert_project_structure_exists(project_dir, "Created project")
