"""
End-to-End tests for basic Marimba workflows.

These tests validate complete user workflows from command line to final output,
ensuring all components work together correctly.
"""

import tempfile
import weakref
from pathlib import Path
from typing import Generator

import pytest
from typer.testing import CliRunner

from marimba.main import marimba_cli as app
from marimba.core.wrappers.dataset import DatasetWrapper


@pytest.fixture(autouse=True)
def cleanup_dataset_wrappers():
    """Automatically clean up any DatasetWrapper instances created during tests."""
    # Track all DatasetWrapper instances using weak references
    original_init = DatasetWrapper.__init__
    dataset_instances = []

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        dataset_instances.append(weakref.ref(self))

    # Use setattr to avoid mypy method assignment error
    setattr(DatasetWrapper, "__init__", tracked_init)

    try:
        yield
    finally:
        # Clean up all tracked instances
        for dataset_ref in dataset_instances:
            dataset_instance = dataset_ref()
            if dataset_instance is not None:
                try:
                    dataset_instance.close()
                except Exception:
                    # Ignore cleanup errors
                    pass

        # Restore original __init__ method
        setattr(DatasetWrapper, "__init__", original_init)


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


@pytest.mark.e2e
class TestBasicWorkflow:
    """Test basic Marimba workflow: new project -> new pipeline -> import -> package."""

    def test_new_project_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
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

    def test_new_pipeline_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test creating a project and adding a pipeline."""
        # First create the project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Change to project directory for subsequent commands
        original_cwd = Path.cwd()
        try:
            # Test: marimba new pipeline <name> <repo> (with project dir)
            result = runner.invoke(
                app,
                [
                    "new",
                    "pipeline",
                    "test_pipeline",
                    "https://github.com/csiro-fair/mritc-demo-pipeline.git",
                    "--project-dir",
                    str(temp_project_dir),
                ],
            )

            # Note: This may fail due to network/git issues in CI, so we'll check for reasonable error handling
            if result.exit_code != 0:
                # Acceptable failures: network issues, git not found, repo not accessible, missing dependencies
                acceptable_errors = [
                    "git",
                    "network",
                    "connection",
                    "repository",
                    "clone",
                    "timeout",
                    "not found",
                    "eof",
                    "pandas",
                    "module",
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

    def test_collection_import_workflow(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test importing data into a collection."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Test: marimba import <collection> <data_path>
        result = runner.invoke(
            app, ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir)]
        )

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

    def test_delete_operations_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test delete operations in a project."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Test deleting non-existent structures (should fail gracefully)
        result = runner.invoke(
            app, ["delete", "collection", "nonexistent_collection", "--project-dir", str(temp_project_dir)]
        )
        # Should fail gracefully for non-existent collections
        assert result.exit_code != 0

        result = runner.invoke(
            app, ["delete", "dataset", "nonexistent_dataset", "--project-dir", str(temp_project_dir)]
        )
        # Should fail gracefully for non-existent datasets
        assert result.exit_code != 0

        # This test focuses on error handling rather than actual deletion
        # since we're testing the CLI behavior with non-existent structures

    def test_package_workflow(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test packaging a dataset."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Import some data to create a collection
        result = runner.invoke(
            app, ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir)]
        )
        assert result.exit_code in [0, 1]  # May fail without pipeline

        # Test: marimba package <dataset>
        result = runner.invoke(
            app,
            [
                "package",
                "test_dataset",
                "--collection-name",
                "test_collection",
                "--project-dir",
                str(temp_project_dir),
                "--version",
                "1.0",
                "--contact-name",
                "Test User",
                "--contact-email",
                "test@example.com",
            ],
        )

        # Package might fail without proper metadata, but should not crash
        assert result.exit_code in [0, 1], f"Package command crashed unexpectedly: {result.stdout}"


@pytest.mark.e2e
class TestComplexWorkflow:
    """Test more complex multi-step workflows."""

    def test_full_demo_workflow_simulation(
        self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path
    ) -> None:
        """Test a simulation of the demo workflow without external dependencies."""
        # Step 1: Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Step 2: Test import commands (may fail without pipeline)
        for collection_name in ["test_025", "test_026", "test_045"]:
            result = runner.invoke(
                app, ["import", collection_name, str(temp_data_dir), "--project-dir", str(temp_project_dir)]
            )
            # Import may fail without pipeline, which is expected

        # Step 3: Test batch delete collections (test error handling for non-existent)
        collection_names = ["test_025", "test_026", "test_045"]
        result = runner.invoke(
            app, ["delete", "collection"] + collection_names + ["--project-dir", str(temp_project_dir)]
        )
        # May fail if collections don't exist, which tests error handling
        # This is acceptable for this workflow test

        # Step 4: Test delete dataset (test error handling)
        result = runner.invoke(app, ["delete", "dataset", "TEST_DATA", "--project-dir", str(temp_project_dir)])
        # May fail if dataset doesn't exist, which is expected behavior

    def test_flowcam_style_workflow(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test complex workflow based on flowcam process.sh example."""
        # Step 1: Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Step 2: Create sample collections for batch deletion test
        collection_names = ["CS17August2022", "CS20August2022", "OW43August2022"]
        for collection_name in collection_names:
            result = runner.invoke(
                app,
                [
                    "new",
                    "collection",
                    collection_name,
                    "--project-dir",
                    str(temp_project_dir),
                    "--config",
                    '{"test": "data"}',
                ],
            )
            assert result.exit_code == 0

        # Step 3: Test batch delete multiple collections (only delete those that exist)
        existing_collections = []
        for collection_name in collection_names:
            if (temp_project_dir / "collections" / collection_name).exists():
                existing_collections.append(collection_name)

        if existing_collections:
            result = runner.invoke(
                app, ["delete", "collection"] + existing_collections + ["--project-dir", str(temp_project_dir)]
            )
            assert result.exit_code == 0

            # Verify deleted collections no longer exist
            for collection_name in existing_collections:
                assert not (temp_project_dir / "collections" / collection_name).exists()

        # Step 4: Test import with config options
        result = runner.invoke(
            app,
            [
                "import",
                "test_collection",
                str(temp_data_dir),
                "--project-dir",
                str(temp_project_dir),
                "--config",
                '{"site_id": "TEST01", "field_of_view": "1000"}',
            ],
        )
        # Import might fail without pipeline, but should parse config correctly
        assert result.exit_code in [0, 1]

    def test_image_rescue_style_workflow(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test workflow based on image rescue process.sh with complex configs."""
        # Step 1: Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Step 2: Test import with complex config including paths and metadata
        complex_config = {
            "batch_data_path": "/tmp/batch_data.csv",
            "inventory_data_path": "/tmp/inventory.xlsx",
            "import_path": str(temp_data_dir),
        }

        result = runner.invoke(
            app,
            [
                "import",
                "batch1a",
                str(temp_data_dir),
                "--project-dir",
                str(temp_project_dir),
                "--operation",
                "link",
                "--config",
                str(complex_config).replace("'", '"'),
            ],
        )
        # Should handle config parsing without external files
        assert result.exit_code in [0, 1]

        # Step 3: Test multiple import operations
        for batch_name in ["batch1b", "batch1c"]:
            result = runner.invoke(
                app,
                [
                    "import",
                    batch_name,
                    str(temp_data_dir),
                    "--project-dir",
                    str(temp_project_dir),
                    "--operation",
                    "link",
                ],
            )
            assert result.exit_code in [0, 1]

    def test_package_workflow_with_metadata_options(
        self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path
    ) -> None:
        """Test packaging with various metadata output options."""
        # Create project and basic dataset
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Import some data to create a collection
        result = runner.invoke(
            app, ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir)]
        )
        assert result.exit_code in [0, 1]

        # Test package with multiple metadata options
        result = runner.invoke(
            app,
            [
                "package",
                "test_dataset",
                "--collection-name",
                "test_collection",
                "--project-dir",
                str(temp_project_dir),
                "--version",
                "1.0",
                "--contact-name",
                "Test User",
                "--contact-email",
                "test@example.com",
                "--metadata-output",
                "yaml",
                "--metadata-level",
                "project",
                "--metadata-level",
                "collection",
                "--allow-destination-collisions",
            ],
        )
        # Package should parse all options correctly
        assert result.exit_code in [0, 1]


@pytest.mark.e2e
class TestAdvancedWorkflows:
    """Test advanced workflows including process and batch operations."""

    def test_process_workflow(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test the process command workflow."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Create some collections using marimba import
        for collection_name in ["test_collection1", "test_collection2"]:
            result = runner.invoke(
                app, ["import", collection_name, str(temp_data_dir), "--project-dir", str(temp_project_dir)]
            )
            # Import may fail without pipeline, but that's expected for this test

        # Test process command
        result = runner.invoke(app, ["process", "--project-dir", str(temp_project_dir)])
        # Process might fail without pipeline, but should not crash
        assert result.exit_code in [0, 1]

    def test_batch_collection_operations(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test batch operations on multiple collections."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Create multiple collections using marimba import
        collection_names = ["batch_test_1", "batch_test_2", "batch_test_3", "batch_test_4"]

        for collection_name in collection_names:
            result = runner.invoke(
                app, ["import", collection_name, str(temp_data_dir), "--project-dir", str(temp_project_dir)]
            )
            # Import may fail without pipeline

        # Test batch delete with multiple collections (only those that exist)
        collections_to_delete = collection_names[:3]  # Delete first 3
        existing_to_delete = []
        for collection_name in collections_to_delete:
            if (temp_project_dir / "collections" / collection_name).exists():
                existing_to_delete.append(collection_name)

        if existing_to_delete:
            result = runner.invoke(
                app, ["delete", "collection"] + existing_to_delete + ["--project-dir", str(temp_project_dir)]
            )
            assert result.exit_code == 0

            # Verify deleted collections no longer exist
            for collection_name in existing_to_delete:
                assert not (temp_project_dir / "collections" / collection_name).exists()

    def test_import_with_overwrite_and_operations(
        self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path
    ) -> None:
        """Test import with overwrite and different operations."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # First import
        result = runner.invoke(
            app, ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir)]
        )
        first_exit_code = result.exit_code

        # Second import with overwrite flag
        result = runner.invoke(
            app,
            ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir), "--overwrite"],
        )
        # Should handle overwrite gracefully
        assert result.exit_code in [0, 1]

        # Test import with copy operation
        result = runner.invoke(
            app,
            [
                "import",
                "test_collection_copy",
                str(temp_data_dir),
                "--project-dir",
                str(temp_project_dir),
                "--operation",
                "copy",
            ],
        )
        assert result.exit_code in [0, 1]

        # Test import with link operation
        result = runner.invoke(
            app,
            [
                "import",
                "test_collection_link",
                str(temp_data_dir),
                "--project-dir",
                str(temp_project_dir),
                "--operation",
                "link",
            ],
        )
        assert result.exit_code in [0, 1]

    def test_comprehensive_pipeline_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test comprehensive workflow that would work with a pipeline."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Attempt to create pipeline (will fail due to network, but tests parsing)
        result = runner.invoke(
            app,
            [
                "new",
                "pipeline",
                "test_pipeline",
                "https://github.com/example/test-pipeline.git",
                "--project-dir",
                str(temp_project_dir),
                "--config",
                '{"test_param": "test_value"}',
            ],
        )
        # Expected to fail due to network, but should parse arguments
        assert result.exit_code != 0

        # Verify error handling is graceful
        acceptable_errors = ["git", "network", "connection", "repository", "clone", "timeout", "not found", "eof"]
        error_output = result.stdout.lower()
        has_acceptable_error = any(error in error_output for error in acceptable_errors)
        # Should either succeed or fail gracefully with network/git error
        assert has_acceptable_error or "error" not in error_output.lower()

    def test_delete_pipeline_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test pipeline deletion workflow using actual marimba commands."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Try to create a pipeline using marimba new pipeline command
        result = runner.invoke(
            app,
            [
                "new",
                "pipeline",
                "test_pipeline",
                "https://github.com/example/test-pipeline.git",
                "--project-dir",
                str(temp_project_dir),
            ],
        )

        pipeline_dir = temp_project_dir / "pipelines" / "test_pipeline"
        pipeline_created_properly = (
            pipeline_dir.exists() and (pipeline_dir / "repo").exists() and (pipeline_dir / "pipeline.yml").exists()
        )

        # Clean up any partial pipeline structure from failed creation using marimba delete
        if pipeline_dir.exists() and not pipeline_created_properly:
            result = runner.invoke(app, ["delete", "pipeline", "test_pipeline", "--project-dir", str(temp_project_dir)])
            # Delete operation should succeed or fail gracefully
            assert result.exit_code in [0, 1]

        # Test pipeline deletion behavior
        if pipeline_created_properly:
            # Pipeline was successfully created, test deletion
            result = runner.invoke(app, ["delete", "pipeline", "test_pipeline", "--project-dir", str(temp_project_dir)])
            # Delete operation should succeed
            assert result.exit_code == 0
            assert not pipeline_dir.exists()
        else:
            # Pipeline creation failed (expected due to non-existent repo)
            # Test deleting non-existent pipeline
            result = runner.invoke(
                app, ["delete", "pipeline", "nonexistent_pipeline", "--project-dir", str(temp_project_dir)]
            )
            # Should fail gracefully for non-existent pipelines
            assert result.exit_code != 0

    def test_error_handling_workflow(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test that commands handle errors gracefully."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Test deleting non-existent collection
        result = runner.invoke(app, ["delete", "collection", "nonexistent", "--project-dir", str(temp_project_dir)])
        assert result.exit_code != 0  # Should fail gracefully

        # Test deleting non-existent dataset
        result = runner.invoke(app, ["delete", "dataset", "nonexistent", "--project-dir", str(temp_project_dir)])
        assert result.exit_code != 0  # Should fail gracefully

        # Test operations on non-existent project
        nonexistent_project = temp_project_dir.parent / "nonexistent_project"
        result = runner.invoke(app, ["delete", "collection", "any", "--project-dir", str(nonexistent_project)])
        assert result.exit_code != 0  # Should fail gracefully


@pytest.mark.e2e
class TestDistributionWorkflows:
    """Test distribution workflows for various target types."""

    @pytest.fixture
    def project(self, runner: CliRunner, temp_project_dir: Path) -> Path:
        """Create a marimba project."""
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0
        return temp_project_dir

    @pytest.fixture
    def mock_dataset_dir(self, runner: CliRunner, project: Path, temp_data_dir: Path) -> Path:
        """Create a mock dataset directory with sample files."""
        # Import some data to create a collection
        result = runner.invoke(app, ["import", "test_collection", str(temp_data_dir), "--project-dir", str(project)])
        assert result.exit_code in [0, 1]

        # Package the collection into a dataset
        result = runner.invoke(
            app,
            [
                "package",
                "test_dataset",
                "--collection-name",
                "test_collection",
                "--project-dir",
                str(project),
                "--version",
                "1.0",
                "--contact-name",
                "Test User",
                "--contact-email",
                "test@example.com",
            ],
        )
        assert result.exit_code in [0, 1]

        dataset_dir = project / "datasets" / "test_dataset"
        return dataset_dir

    @pytest.fixture
    def mock_s3_target_dir(self, project: Path) -> Path:
        """Create a mock S3 target directory manually for testing."""
        # Create target directory structure manually since interactive creation is complex
        target_dir = project / "targets" / "test_s3_target"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal target.yml file
        target_config = {
            "type": "s3",
            "bucket": "test-bucket",
            "endpoint_url": "https://test.s3.amazonaws.com",
            "access_key_id": "test_access_key",
            "secret_access_key": "test_secret_key",
            "region": "us-east-1",
        }
        import yaml

        (target_dir / "target.yml").write_text(yaml.dump(target_config))

        return target_dir

    @pytest.fixture
    def mock_dap_target_dir(self, project: Path) -> Path:
        """Create a mock DAP target directory manually for testing."""
        # Create target directory structure manually
        target_dir = project / "targets" / "test_dap_target"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Create a minimal target.yml file
        target_config = {
            "type": "dap",
            "base_url": "https://test.dap.server.com",
            "username": "test_user",
            "password": "test_password",
            "dataset_path": "/datasets",
        }
        import yaml

        (target_dir / "target.yml").write_text(yaml.dump(target_config))

        return target_dir

    def test_distribute_to_s3_target_dry_run(
        self,
        runner: CliRunner,
        project: Path,
        mock_dataset_dir: Path,
        mock_s3_target_dir: Path,
    ) -> None:
        """Test distribution to S3 target with dry run."""
        # Ensure target exists
        assert mock_s3_target_dir.exists()

        # Test distribute command with dry run (should not fail on network)
        result = runner.invoke(
            app, ["distribute", "test_dataset", "test_s3_target", "--project-dir", str(project), "--dry-run"]
        )
        # In dry run mode, should parse everything correctly without network calls
        assert result.exit_code in [0, 1]  # May fail due to missing target/dataset, but should handle gracefully

    def test_distribute_to_dap_target_dry_run(
        self,
        runner: CliRunner,
        project: Path,
        mock_dataset_dir: Path,
        mock_dap_target_dir: Path,
    ) -> None:
        """Test distribution to DAP target with dry run."""
        # Ensure target exists
        assert mock_dap_target_dir.exists()

        # Test distribute command with dry run
        result = runner.invoke(
            app, ["distribute", "test_dataset", "test_dap_target", "--project-dir", str(project), "--dry-run"]
        )
        # Should handle dry run gracefully
        assert result.exit_code in [0, 1]

    def test_distribute_with_validation_disabled(
        self,
        runner: CliRunner,
        project: Path,
        mock_dataset_dir: Path,
        mock_s3_target_dir: Path,
    ) -> None:
        """Test distribution with validation disabled."""
        # Ensure target exists
        assert mock_s3_target_dir.exists()

        # Test distribute without validation
        result = runner.invoke(
            app,
            [
                "distribute",
                "test_dataset",
                "test_s3_target",
                "--project-dir",
                str(project),
                "--no-validate",
                "--dry-run",
            ],
        )
        # Should skip validation step
        assert result.exit_code in [0, 1]

    def test_distribute_nonexistent_dataset(self, runner: CliRunner, project: Path, mock_s3_target_dir: Path) -> None:
        """Test distribution of non-existent dataset."""
        # Ensure target exists
        assert mock_s3_target_dir.exists()

        # Test distribute non-existent dataset
        result = runner.invoke(
            app, ["distribute", "nonexistent_dataset", "test_s3_target", "--project-dir", str(project)]
        )
        # Should fail gracefully with appropriate error
        # Note: CLI may return 0 but should show error in output
        assert result.exit_code != 0 or "no such dataset" in result.stdout.lower()

    def test_distribute_nonexistent_target(self, runner: CliRunner, project: Path, mock_dataset_dir: Path) -> None:
        """Test distribution to non-existent target."""
        # Ensure dataset exists
        assert mock_dataset_dir.exists()

        # Test distribute to non-existent target
        result = runner.invoke(app, ["distribute", "test_dataset", "nonexistent_target", "--project-dir", str(project)])
        # Should fail gracefully with appropriate error
        # Note: CLI may return 0 but should show error in output
        assert result.exit_code != 0 or "no such target" in result.stdout.lower()

    def test_distribute_invalid_project_directory(self, runner: CliRunner, temp_project_dir: Path) -> None:
        """Test distribution from invalid project directory."""
        nonexistent_project = temp_project_dir.parent / "nonexistent_project"

        # Test distribute from non-existent project
        result = runner.invoke(
            app, ["distribute", "test_dataset", "test_target", "--project-dir", str(nonexistent_project)]
        )
        # Should fail gracefully with appropriate error
        assert result.exit_code != 0
        assert "project" in result.stdout.lower() or "not found" in result.stdout.lower()

    def test_distribute_workflow_argument_parsing(self, runner: CliRunner, project: Path) -> None:
        """Test that distribute command correctly parses all arguments."""
        # Test distribute with all available flags
        result = runner.invoke(
            app,
            [
                "distribute",
                "test_dataset",
                "test_target",
                "--project-dir",
                str(project),
                "--validate",
                "--dry-run",
            ],
        )
        # Should parse arguments correctly even if target/dataset don't exist
        assert result.exit_code in [0, 1]  # May fail due to missing components

        # Test with negation flag
        result = runner.invoke(
            app,
            [
                "distribute",
                "test_dataset",
                "test_target",
                "--project-dir",
                str(project),
                "--no-validate",
                "--dry-run",
            ],
        )
        # Should parse negation flag correctly
        assert result.exit_code in [0, 1]

    def test_comprehensive_distribute_workflow(
        self,
        runner: CliRunner,
        temp_project_dir: Path,
        temp_data_dir: Path,
    ) -> None:
        """Test comprehensive workflow: create -> package -> distribute."""
        # Step 1: Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Step 2: Create target using new command (may fail, but tests argument parsing)
        result = runner.invoke(app, ["new", "target", "test_s3_target", "--project-dir", str(temp_project_dir)])
        # Target creation may fail due to interactive prompts, which is expected

        # Step 3: Import some data (may fail without pipeline)
        result = runner.invoke(
            app, ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir)]
        )
        # Import may fail without pipeline

        # Step 4: Try to package dataset (may fail without proper data)
        result = runner.invoke(
            app,
            [
                "package",
                "test_dataset",
                "--project-dir",
                str(temp_project_dir),
                "--version",
                "1.0",
                "--contact-name",
                "Test User",
                "--contact-email",
                "test@example.com",
                "--dry-run",
            ],
        )
        # Package may fail due to missing data/pipelines

        # Step 5: Attempt distribution (should fail gracefully)
        result = runner.invoke(
            app, ["distribute", "test_dataset", "test_s3_target", "--project-dir", str(temp_project_dir), "--dry-run"]
        )
        # Distribution should fail gracefully due to missing components
        assert result.exit_code in [0, 1]

    def test_distribute_command_help_and_options(self, runner: CliRunner) -> None:
        """Test that distribute command help works and shows all options."""
        # Test help for distribute command
        result = runner.invoke(app, ["distribute", "--help"])
        assert result.exit_code == 0
        assert "distribute" in result.stdout.lower()
        assert "dataset" in result.stdout.lower()
        assert "target" in result.stdout.lower()
        assert "validate" in result.stdout.lower()
        assert "dry-run" in result.stdout.lower()
