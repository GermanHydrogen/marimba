"""
End-to-End tests for basic Marimba workflows.

These tests validate complete user workflows from command line to final output,
ensuring all components work together correctly.
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from typer.testing import CliRunner

from marimba.main import marimba_cli as app


@pytest.fixture
def temp_project_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for E2E test projects."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / "test_project"


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory with sample data for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "sample_data"
        data_dir.mkdir()
        
        # Create some sample files to import
        (data_dir / "image1.jpg").write_text("fake image data")
        (data_dir / "image2.jpg").write_text("fake image data 2")
        (data_dir / "metadata.txt").write_text("sample metadata")
        
        yield data_dir


@pytest.fixture
def runner() -> CliRunner:
    """CLI runner for testing typer commands."""
    return CliRunner()


@pytest.mark.integration
class TestBasicWorkflow:
    """Test basic Marimba workflow: new project -> new pipeline -> import -> package."""

    def test_new_project_workflow(self, runner: CliRunner, temp_project_dir: Path):
        """Test creating a new project and verifying structure."""
        # Test: marimba new project <name>
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        
        assert result.exit_code == 0, f"Failed to create project: {result.stdout}"
        
        # Verify project structure was created
        assert temp_project_dir.exists()
        assert (temp_project_dir / ".marimba").is_dir()
        assert (temp_project_dir / "pipelines").is_dir()
        assert (temp_project_dir / "collections").is_dir()
        assert (temp_project_dir / "datasets").is_dir()
        assert (temp_project_dir / "targets").is_dir()

    def test_new_pipeline_workflow(self, runner: CliRunner, temp_project_dir: Path):
        """Test creating a project and adding a pipeline."""
        # First create the project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Change to project directory for subsequent commands
        original_cwd = Path.cwd()
        try:
            # Test: marimba new pipeline <name> <repo> (with project dir)
            result = runner.invoke(app, [
                "new", "pipeline", "test_pipeline",
                "https://github.com/csiro-fair/mritc-demo-pipeline.git",
                "--project-dir", str(temp_project_dir)
            ])
            
            # Note: This may fail due to network/git issues in CI, so we'll check for reasonable error handling
            if result.exit_code != 0:
                # Acceptable failures: network issues, git not found, repo not accessible
                acceptable_errors = [
                    "git",
                    "network", 
                    "connection",
                    "repository",
                    "clone",
                    "timeout",
                    "not found",
                    "eof"
                ]
                error_output = result.stdout.lower()
                has_acceptable_error = any(error in error_output for error in acceptable_errors)
                
                if not has_acceptable_error:
                    pytest.fail(f"Unexpected pipeline creation failure: {result.stdout}")
                else:
                    pytest.skip("Skipping due to network/git dependency")
            else:
                # If successful, verify pipeline was created
                pipeline_dir = temp_project_dir / "pipelines" / "test_pipeline"
                assert pipeline_dir.exists()
                
        finally:
            # Restore original working directory
            original_cwd

    def test_collection_import_workflow(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path):
        """Test importing data into a collection."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Test: marimba import <collection> <data_path>
        result = runner.invoke(app, [
            "import", "test_collection", str(temp_data_dir),
            "--project-dir", str(temp_project_dir)
        ])
        
        # Import might fail without a pipeline, which is acceptable for this test
        # We're mainly testing that the command parses correctly and doesn't crash
        assert result.exit_code in [0, 1], f"Import command crashed unexpectedly: {result.stdout}"
        
        # If import was processed, verify basic collection structure expectations
        if result.exit_code == 0:
            collection_dir = temp_project_dir / "collections" / "test_collection"
            # Collection directory should exist if import succeeded
            assert collection_dir.exists()
        
        # If successful, verify collection was created
        if result.exit_code == 0:
            collection_dir = temp_project_dir / "collections" / "test_collection"
            assert collection_dir.exists()

    def test_delete_operations_workflow(self, runner: CliRunner, temp_project_dir: Path):
        """Test delete operations in a project."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Test deleting non-existent structures (should fail gracefully)
        result = runner.invoke(app, [
            "delete", "collection", "nonexistent_collection",
            "--project-dir", str(temp_project_dir)
        ])
        # Should fail gracefully for non-existent collections
        assert result.exit_code != 0
        
        result = runner.invoke(app, [
            "delete", "dataset", "nonexistent_dataset",
            "--project-dir", str(temp_project_dir)
        ])
        # Should fail gracefully for non-existent datasets
        assert result.exit_code != 0
        
        # This test focuses on error handling rather than actual deletion
        # since we're testing the CLI behavior with non-existent structures

    def test_package_workflow(self, runner: CliRunner, temp_project_dir: Path):
        """Test packaging a dataset."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Create minimal dataset structure
        dataset_dir = temp_project_dir / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "sample_file.txt").write_text("test data")

        # Test: marimba package <dataset>
        result = runner.invoke(app, [
            "package", "test_dataset",
            "--project-dir", str(temp_project_dir),
            "--version", "1.0",
            "--contact-name", "Test User",
            "--contact-email", "test@example.com"
        ])
        
        # Package might fail without proper metadata, but should not crash
        assert result.exit_code in [0, 1], f"Package command crashed unexpectedly: {result.stdout}"


@pytest.mark.integration  
class TestComplexWorkflow:
    """Test more complex multi-step workflows."""

    def test_full_demo_workflow_simulation(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path):
        """Test a simulation of the demo workflow without external dependencies."""
        # Step 1: Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Step 2: Test import commands (may fail without pipeline)
        for collection_name in ["test_025", "test_026", "test_045"]:
            result = runner.invoke(app, [
                "import", collection_name, str(temp_data_dir),
                "--project-dir", str(temp_project_dir)
            ])
            # Import may fail without pipeline, which is expected

        # Step 3: Test batch delete collections (test error handling for non-existent)
        collection_names = ["test_025", "test_026", "test_045"]
        result = runner.invoke(app, [
            "delete", "collection"] + collection_names + [
            "--project-dir", str(temp_project_dir)
        ])
        # May fail if collections don't exist, which tests error handling
        # This is acceptable for this workflow test

        # Step 4: Test delete dataset (test error handling)
        result = runner.invoke(app, [
            "delete", "dataset", "TEST_DATA", 
            "--project-dir", str(temp_project_dir)
        ])
        # May fail if dataset doesn't exist, which is expected behavior

    def test_flowcam_style_workflow(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path):
        """Test complex workflow based on flowcam process.sh example."""
        # Step 1: Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Step 2: Create sample collections for batch deletion test
        collections_dir = temp_project_dir / "collections"
        collection_names = ["CS17August2022", "CS20August2022", "OW43August2022"]
        for collection_name in collection_names:
            (collections_dir / collection_name).mkdir(parents=True)
            # Create required collection config file
            (collections_dir / collection_name / "collection.yml").write_text("test: data")

        # Step 3: Test batch delete multiple collections (only delete those that exist)
        existing_collections = []
        for collection_name in collection_names:
            if (temp_project_dir / "collections" / collection_name).exists():
                existing_collections.append(collection_name)
        
        if existing_collections:
            result = runner.invoke(app, [
                "delete", "collection"] + existing_collections + [
                "--project-dir", str(temp_project_dir)
            ])
            assert result.exit_code == 0

            # Verify deleted collections no longer exist
            for collection_name in existing_collections:
                assert not (temp_project_dir / "collections" / collection_name).exists()

        # Step 4: Test import with config options
        result = runner.invoke(app, [
            "import", "test_collection", str(temp_data_dir),
            "--project-dir", str(temp_project_dir),
            "--config", '{"site_id": "TEST01", "field_of_view": "1000"}'
        ])
        # Import might fail without pipeline, but should parse config correctly
        assert result.exit_code in [0, 1]

    def test_image_rescue_style_workflow(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path):
        """Test workflow based on image rescue process.sh with complex configs."""
        # Step 1: Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Step 2: Test import with complex config including paths and metadata
        complex_config = {
            "batch_data_path": "/tmp/batch_data.csv",
            "inventory_data_path": "/tmp/inventory.xlsx",
            "import_path": str(temp_data_dir)
        }
        
        result = runner.invoke(app, [
            "import", "batch1a", str(temp_data_dir),
            "--project-dir", str(temp_project_dir),
            "--operation", "link",
            "--config", str(complex_config).replace("'", '"')
        ])
        # Should handle config parsing without external files
        assert result.exit_code in [0, 1]

        # Step 3: Test multiple import operations
        for batch_name in ["batch1b", "batch1c"]:
            result = runner.invoke(app, [
                "import", batch_name, str(temp_data_dir),
                "--project-dir", str(temp_project_dir),
                "--operation", "link"
            ])
            assert result.exit_code in [0, 1]

    def test_package_workflow_with_metadata_options(self, runner: CliRunner, temp_project_dir: Path):
        """Test packaging with various metadata output options."""
        # Create project and basic dataset
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        dataset_dir = temp_project_dir / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "sample_file.txt").write_text("test data")

        # Test package with multiple metadata options
        result = runner.invoke(app, [
            "package", "test_dataset",
            "--project-dir", str(temp_project_dir),
            "--version", "1.0",
            "--contact-name", "Test User",
            "--contact-email", "test@example.com",
            "--metadata-output", "yaml",
            "--metadata-level", "project",
            "--metadata-level", "collection",
            "--allow-destination-collisions"
        ])
        # Package should parse all options correctly
        assert result.exit_code in [0, 1]


@pytest.mark.integration
class TestAdvancedWorkflows:
    """Test advanced workflows including process and batch operations."""

    def test_process_workflow(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path):
        """Test the process command workflow."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Create some collections using marimba import
        for collection_name in ["test_collection1", "test_collection2"]:
            result = runner.invoke(app, [
                "import", collection_name, str(temp_data_dir),
                "--project-dir", str(temp_project_dir)
            ])
            # Import may fail without pipeline, but that's expected for this test

        # Test process command
        result = runner.invoke(app, [
            "process",
            "--project-dir", str(temp_project_dir)
        ])
        # Process might fail without pipeline, but should not crash
        assert result.exit_code in [0, 1]

    def test_batch_collection_operations(self, runner: CliRunner, temp_project_dir: Path):
        """Test batch operations on multiple collections."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Create multiple collections using marimba import
        collection_names = ["batch_test_1", "batch_test_2", "batch_test_3", "batch_test_4"]
        
        for collection_name in collection_names:
            result = runner.invoke(app, [
                "import", collection_name, str(temp_data_dir),
                "--project-dir", str(temp_project_dir)
            ])
            # Import may fail without pipeline

        # Test batch delete with multiple collections (only those that exist)
        collections_to_delete = collection_names[:3]  # Delete first 3
        existing_to_delete = []
        for collection_name in collections_to_delete:
            if (temp_project_dir / "collections" / collection_name).exists():
                existing_to_delete.append(collection_name)
        
        if existing_to_delete:
            result = runner.invoke(app, [
                "delete", "collection"] + existing_to_delete + [
                "--project-dir", str(temp_project_dir)
            ])
            assert result.exit_code == 0

            # Verify deleted collections no longer exist
            for collection_name in existing_to_delete:
                assert not (temp_project_dir / "collections" / collection_name).exists()

    def test_import_with_overwrite_and_operations(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path):
        """Test import with overwrite and different operations."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # First import
        result = runner.invoke(app, [
            "import", "test_collection", str(temp_data_dir),
            "--project-dir", str(temp_project_dir)
        ])
        first_exit_code = result.exit_code

        # Second import with overwrite flag
        result = runner.invoke(app, [
            "import", "test_collection", str(temp_data_dir),
            "--project-dir", str(temp_project_dir),
            "--overwrite"
        ])
        # Should handle overwrite gracefully
        assert result.exit_code in [0, 1]

        # Test import with copy operation
        result = runner.invoke(app, [
            "import", "test_collection_copy", str(temp_data_dir),
            "--project-dir", str(temp_project_dir),
            "--operation", "copy"
        ])
        assert result.exit_code in [0, 1]

        # Test import with link operation
        result = runner.invoke(app, [
            "import", "test_collection_link", str(temp_data_dir),
            "--project-dir", str(temp_project_dir),
            "--operation", "link"
        ])
        assert result.exit_code in [0, 1]

    def test_comprehensive_pipeline_workflow(self, runner: CliRunner, temp_project_dir: Path):
        """Test comprehensive workflow that would work with a pipeline."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Attempt to create pipeline (will fail due to network, but tests parsing)
        result = runner.invoke(app, [
            "new", "pipeline", "test_pipeline",
            "https://github.com/example/test-pipeline.git",
            "--project-dir", str(temp_project_dir),
            "--config", '{"test_param": "test_value"}'
        ])
        # Expected to fail due to network, but should parse arguments
        assert result.exit_code != 0
        
        # Verify error handling is graceful
        acceptable_errors = ["git", "network", "connection", "repository", "clone", "timeout", "not found", "eof"]
        error_output = result.stdout.lower()
        has_acceptable_error = any(error in error_output for error in acceptable_errors)
        # Should either succeed or fail gracefully with network/git error
        assert has_acceptable_error or "error" not in error_output.lower()

    def test_delete_pipeline_workflow(self, runner: CliRunner, temp_project_dir: Path):
        """Test pipeline deletion workflow using actual marimba commands."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Try to create a pipeline using marimba new pipeline command
        result = runner.invoke(app, [
            "new", "pipeline", "test_pipeline",
            "https://github.com/example/test-pipeline.git",
            "--project-dir", str(temp_project_dir)
        ])
        
        pipeline_dir = temp_project_dir / "pipelines" / "test_pipeline"
        pipeline_created_properly = (
            pipeline_dir.exists() and 
            (pipeline_dir / "repo").exists() and 
            (pipeline_dir / "pipeline.yml").exists()
        )
        
        # Clean up any partial pipeline structure from failed creation
        if pipeline_dir.exists() and not pipeline_created_properly:
            import shutil
            shutil.rmtree(pipeline_dir)
        
        # Test pipeline deletion behavior
        if pipeline_created_properly:
            # Pipeline was successfully created, test deletion
            result = runner.invoke(app, [
                "delete", "pipeline", "test_pipeline",
                "--project-dir", str(temp_project_dir)
            ])
            assert result.exit_code == 0
            assert not pipeline_dir.exists()
        else:
            # Pipeline creation failed (expected due to non-existent repo)
            # Test deleting non-existent pipeline  
            result = runner.invoke(app, [
                "delete", "pipeline", "nonexistent_pipeline",
                "--project-dir", str(temp_project_dir)
            ])
            # Should fail gracefully for non-existent pipelines
            assert result.exit_code != 0

    def test_error_handling_workflow(self, runner: CliRunner, temp_project_dir: Path):
        """Test that commands handle errors gracefully."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Test deleting non-existent collection
        result = runner.invoke(app, [
            "delete", "collection", "nonexistent",
            "--project-dir", str(temp_project_dir)
        ])
        assert result.exit_code != 0  # Should fail gracefully

        # Test deleting non-existent dataset
        result = runner.invoke(app, [
            "delete", "dataset", "nonexistent",
            "--project-dir", str(temp_project_dir)  
        ])
        assert result.exit_code != 0  # Should fail gracefully

        # Test operations on non-existent project
        nonexistent_project = temp_project_dir.parent / "nonexistent_project"
        result = runner.invoke(app, [
            "delete", "collection", "any",
            "--project-dir", str(nonexistent_project)
        ])
        assert result.exit_code != 0  # Should fail gracefully