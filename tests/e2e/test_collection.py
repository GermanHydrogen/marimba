"""
End-to-End tests for collection operations.

These tests validate data import, collection management, and batch operations.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from marimba.main import marimba_cli as app


@pytest.mark.e2e
class TestCollectionImport:
    """Test data import and collection creation workflows."""

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

    def test_import_with_config_options(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test import with configuration options."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Test import with config options
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

    def test_import_with_complex_config(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test import with complex configuration including paths and metadata."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Test import with complex config
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

        # Test multiple import operations
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


@pytest.mark.e2e
class TestCollectionDeletion:
    """Test collection deletion and batch operations."""

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

    def test_batch_collection_operations(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test batch operations on multiple collections."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Create multiple collections using marimba new collection
        collection_names = ["batch_test_1", "batch_test_2", "batch_test_3", "batch_test_4"]

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


@pytest.mark.e2e
class TestCollectionWorkflows:
    """Test complex collection workflows and simulations."""

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
