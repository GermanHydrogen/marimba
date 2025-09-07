"""Tests for marimba.core.utils.rich module."""

import pytest
from rich.panel import Panel

from marimba.core.utils.rich import (
    MARIMBA,
    error_panel,
    format_command,
    format_entity,
    get_default_columns,
    success_panel,
    warning_panel,
)


class TestRichUtils:
    """Test rich utility functions."""

    @pytest.mark.unit
    def test_success_panel(self):
        """Test success_panel creates proper panel."""
        result = success_panel("Test message")

        assert isinstance(result, Panel)
        assert "Test message" in str(result.renderable)
        assert result.title == "Success"
        assert result.border_style == "green"

    @pytest.mark.unit
    def test_success_panel_custom_title(self):
        """Test success_panel with custom title."""
        result = success_panel("Test message", title="Custom Success")

        assert result.title == "Custom Success"

    @pytest.mark.unit
    def test_warning_panel(self):
        """Test warning_panel creates proper panel."""
        result = warning_panel("Warning message")

        assert isinstance(result, Panel)
        assert "Warning message" in str(result.renderable)
        assert result.title == "Warning"
        assert result.border_style == "yellow"

    @pytest.mark.unit
    def test_warning_panel_custom_title(self):
        """Test warning_panel with custom title."""
        result = warning_panel("Warning message", title="Custom Warning")

        assert result.title == "Custom Warning"

    @pytest.mark.unit
    def test_error_panel(self):
        """Test error_panel creates proper panel."""
        result = error_panel("Error message")

        assert isinstance(result, Panel)
        assert "Error message" in str(result.renderable)
        assert result.title == "Error"
        assert result.border_style == "red"

    @pytest.mark.unit
    def test_error_panel_custom_title(self):
        """Test error_panel with custom title."""
        result = error_panel("Error message", title="Custom Error")

        assert result.title == "Custom Error"

    @pytest.mark.unit
    def test_format_command(self):
        """Test format_command applies steel blue styling."""
        result = format_command("test-command")

        assert result == "[steel_blue3]test-command[/steel_blue3]"

    @pytest.mark.unit
    def test_format_entity(self):
        """Test format_entity applies light pink styling."""
        result = format_entity("test-entity")

        assert result == "[light_pink3]test-entity[/light_pink3]"

    @pytest.mark.unit
    def test_get_default_columns(self):
        """Test get_default_columns returns expected column types."""
        columns = get_default_columns()

        assert len(columns) == 5
        # Check that we get the expected column types
        column_types = [type(col).__name__ for col in columns]
        expected_types = ["TextColumn", "BarColumn", "TaskProgressColumn", "TimeRemainingColumn", "TimeElapsedColumn"]
        assert column_types == expected_types

    @pytest.mark.unit
    def test_marimba_constant(self):
        """Test MARIMBA constant is properly formatted."""
        assert MARIMBA == "[bold][aquamarine3]Marimba[/aquamarine3][/bold]"
