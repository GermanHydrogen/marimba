"""
End-to-End tests for collection operations.

These tests validate data import, collection management, and batch operations.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from marimba.main import marimba_cli as app
from tests.conftest import (
    TestDataFactory,
    assert_cli_success,
)


@pytest.mark.e2e
class TestCollectionImport:
    """Test data import and collection creation workflows."""

    def test_collection_import_workflow(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test importing data into a collection (import succeeds without pipeline, but no processing occurs)."""
        # Create project first - using shared CLI helper
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert_cli_success(result, expected_message="Created new Marimba project", context="Project creation")

        # Test: marimba import <collection> <data_path>
        result = runner.invoke(
            app,
            ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir)],
        )

        # Import should succeed (copies files but doesn't process them without pipelines) - using shared CLI helper
        assert_cli_success(result, context="Collection import")

        # Verify collection directory was created using enhanced validation
        collection_dir = temp_project_dir / "collections" / "test_collection"
        assert collection_dir.exists(), "Collection directory should be created"

        # Verify collection config was created (this is what actually gets created)
        collection_config = collection_dir / "collection.yml"
        assert collection_config.exists(), "Collection config should be created"

        # The import command creates the collection structure but may not copy files
        # without proper pipeline configuration, so we just verify the basic structure

    def test_import_with_config_options(
        self,
        runner: CliRunner,
        temp_project_dir: Path,
        temp_data_dir: Path,
        test_data_factory: TestDataFactory,
    ) -> None:
        """Test import with configuration options (should succeed and properly parse config)."""
        # Create project first
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Use factory to generate test config instead of hard-coded values
        collection_config_data = test_data_factory.create_collection_config(
            site_id="FACTORY_SITE_01",
            field_of_view="1500",
        )
        config_json = str(
            {"site_id": collection_config_data["site_id"], "field_of_view": collection_config_data["field_of_view"]},
        ).replace("'", '"')

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
                config_json,
            ],
        )

        # Should succeed and parse config correctly
        assert result.exit_code == 0, f"Import with config should succeed: {result.stdout}"

        # Verify collection was created with config
        collection_dir = temp_project_dir / "collections" / "test_collection"
        assert collection_dir.exists(), "Collection directory should be created"

        # Verify config was saved
        collection_config = collection_dir / "collection.yml"
        assert collection_config.exists(), "Collection config should be created"

        # Read and verify config contains the factory-generated values
        config_content = collection_config.read_text()
        assert "FACTORY_SITE_01" in config_content, "Config should contain factory-generated site_id value"
        assert "1500" in config_content, "Config should contain factory-generated field_of_view value"

    def test_import_with_overwrite_and_operations(
        self,
        runner: CliRunner,
        temp_project_dir: Path,
        temp_data_dir: Path,
    ) -> None:
        """Test import with overwrite and different operations."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # First import
        result = runner.invoke(
            app,
            ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir)],
        )

        # Second import with overwrite flag (should succeed)
        result = runner.invoke(
            app,
            ["import", "test_collection", str(temp_data_dir), "--project-dir", str(temp_project_dir), "--overwrite"],
        )
        # Should succeed and overwrite existing collection
        assert result.exit_code == 0, f"Import with overwrite should succeed: {result.stdout}"

        # Test import with copy operation (should succeed)
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
        assert result.exit_code == 0, f"Import with copy operation should succeed: {result.stdout}"

        # Verify copy collection was created
        copy_collection_dir = temp_project_dir / "collections" / "test_collection_copy"
        assert copy_collection_dir.exists(), "Copy collection directory should be created"

        # Test import with link operation (should succeed)
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
        assert result.exit_code == 0, f"Import with link operation should succeed: {result.stdout}"

        # Verify link collection was created
        link_collection_dir = temp_project_dir / "collections" / "test_collection_link"
        assert link_collection_dir.exists(), "Link collection directory should be created"

    def test_import_with_complex_config(self, runner: CliRunner, temp_project_dir: Path, temp_data_dir: Path) -> None:
        """Test import with complex configuration parsing."""
        # Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Test import with complex config - this test focuses on JSON parsing
        complex_config = {
            "site_id": "COMPLEX_SITE_01",
            "field_of_view": "2000",
            "instrument_type": "flowcam",
            "depth_range": {"min": 5.0, "max": 25.0},
            "metadata": {"operator": "test_user", "mission": "test_mission_2024"},
        }

        result = runner.invoke(
            app,
            [
                "import",
                "batch_complex",
                str(temp_data_dir),
                "--project-dir",
                str(temp_project_dir),
                "--operation",
                "link",
                "--config",
                str(complex_config).replace("'", '"'),
            ],
        )
        # Should succeed - complex config should parse correctly
        assert result.exit_code == 0, f"Import with complex config should succeed: {result.stdout}"

        # Verify the collection was created
        collection_dir = temp_project_dir / "collections" / "batch_complex"
        assert collection_dir.exists(), "Collection directory should be created"

        # Verify complex config was saved
        collection_config = collection_dir / "collection.yml"
        assert collection_config.exists(), "Collection config should be created"
        config_content = collection_config.read_text()
        assert "COMPLEX_SITE_01" in config_content, "Config should contain complex site_id"
        assert "2000" in config_content, "Config should contain field_of_view"
        assert "test_user" in config_content, "Config should contain nested metadata"


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
            app,
            ["delete", "collection", "nonexistent_collection", "--project-dir", str(temp_project_dir)],
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
        existing_to_delete = [
            collection_name
            for collection_name in collections_to_delete
            if (temp_project_dir / "collections" / collection_name).exists()
        ]

        if existing_to_delete:
            result = runner.invoke(
                app,
                ["delete", "collection", *existing_to_delete, "--project-dir", str(temp_project_dir)],
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
            assert result.exit_code == 0, f"Failed to create collection {collection_name}: {result.stdout}"

        # Step 3: Verify collections were created
        for collection_name in collection_names:
            collection_dir = temp_project_dir / "collections" / collection_name
            assert collection_dir.exists(), f"Collection {collection_name} directory should exist"

        # Step 4: Test batch delete multiple collections
        result = runner.invoke(
            app,
            ["delete", "collection", *collection_names, "--project-dir", str(temp_project_dir)],
        )
        assert result.exit_code == 0, f"Batch collection deletion failed: {result.stdout}"

        # Verify deleted collections no longer exist
        for collection_name in collection_names:
            collection_dir = temp_project_dir / "collections" / collection_name
            assert not collection_dir.exists(), f"Collection {collection_name} should be deleted"

        # Step 5: Test import with config options (should succeed)
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
        # Import should succeed - config parsing works without pipelines
        assert result.exit_code == 0, f"Import with config should succeed: {result.stdout}"

        # Verify collection was created
        test_collection_dir = temp_project_dir / "collections" / "test_collection"
        assert test_collection_dir.exists(), "Test collection should be created"


@pytest.mark.e2e
class TestCollectionWorkflows:
    """Test complex collection workflows and simulations."""

    def test_full_demo_workflow_simulation(
        self,
        runner: CliRunner,
        temp_project_dir: Path,
        temp_data_dir: Path,
    ) -> None:
        """Test a simulation of the demo workflow without external dependencies."""
        # Step 1: Create project
        result = runner.invoke(app, ["new", "project", str(temp_project_dir)])
        assert result.exit_code == 0

        # Step 2: Test import commands (should all succeed)
        imported_collections = []
        for collection_name in ["test_025", "test_026", "test_045"]:
            result = runner.invoke(
                app,
                ["import", collection_name, str(temp_data_dir), "--project-dir", str(temp_project_dir)],
            )
            # All imports should succeed
            assert result.exit_code == 0, f"Import {collection_name} should succeed: {result.stdout}"
            imported_collections.append(collection_name)

            # Verify collection was created
            collection_dir = temp_project_dir / "collections" / collection_name
            assert collection_dir.exists(), f"Collection {collection_name} should be created"

        # Step 3: Test batch delete collections (should succeed now that collections exist)
        collection_names = ["test_025", "test_026", "test_045"]
        result = runner.invoke(
            app,
            ["delete", "collection", *collection_names, "--project-dir", str(temp_project_dir)],
        )
        # Should succeed because collections now exist
        assert result.exit_code == 0, f"Delete should succeed for existing collections: {result.stdout}"

        # Verify collections were deleted
        for collection_name in collection_names:
            collection_dir = temp_project_dir / "collections" / collection_name
            assert not collection_dir.exists(), f"Collection {collection_name} should be deleted"
