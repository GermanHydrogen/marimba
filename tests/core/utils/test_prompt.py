"""
Unit tests for prompt utilities.

Tests the functionality of the prompt_schema function including:
- Prompting for different data types (str, int, float, bool)
- Using default values
- Handling KeyboardInterrupt
- Error handling for unsupported types
"""

from unittest.mock import patch

import pytest

from marimba.core.utils.prompt import prompt_schema


@pytest.mark.unit
class TestPromptSchema:
    """Test prompt_schema function."""

    @patch("rich.prompt.Prompt.ask")
    def test_prompt_string_values(self, mock_ask):
        """Test prompting for string values."""
        schema = {"name": "default_name", "description": "default_desc"}
        mock_ask.side_effect = ["custom_name", "custom_desc"]

        result = prompt_schema(schema)

        assert result == {"name": "custom_name", "description": "custom_desc"}
        assert mock_ask.call_count == 2

    @patch("rich.prompt.IntPrompt.ask")
    def test_prompt_integer_values(self, mock_ask):
        """Test prompting for integer values."""
        schema = {"count": 10, "max_items": 100}
        mock_ask.side_effect = [25, 200]

        result = prompt_schema(schema)

        assert result == {"count": 25, "max_items": 200}
        assert mock_ask.call_count == 2

    @patch("rich.prompt.FloatPrompt.ask")
    def test_prompt_float_values(self, mock_ask):
        """Test prompting for float values."""
        schema = {"threshold": 0.5, "factor": 1.0}
        mock_ask.side_effect = [0.8, 2.5]

        result = prompt_schema(schema)

        assert result == {"threshold": 0.8, "factor": 2.5}
        assert mock_ask.call_count == 2

    @patch("rich.prompt.Confirm.ask")
    def test_prompt_boolean_values(self, mock_ask):
        """Test prompting for boolean values."""
        schema = {"enabled": True, "debug": False}
        mock_ask.side_effect = [False, True]

        result = prompt_schema(schema)

        assert result == {"enabled": False, "debug": True}
        assert mock_ask.call_count == 2

    @patch("rich.prompt.Confirm.ask")
    @patch("rich.prompt.IntPrompt.ask")
    @patch("rich.prompt.FloatPrompt.ask")
    @patch("rich.prompt.Prompt.ask")
    def test_prompt_mixed_types(self, mock_prompt, mock_float, mock_int, mock_confirm):
        """Test prompting for mixed data types."""
        schema = {"name": "test", "count": 5, "ratio": 0.5, "enabled": True}
        mock_prompt.return_value = "new_name"
        mock_int.return_value = 10
        mock_float.return_value = 0.75
        mock_confirm.return_value = False

        result = prompt_schema(schema)

        assert result == {"name": "new_name", "count": 10, "ratio": 0.75, "enabled": False}
        mock_prompt.assert_called_once_with("name", default="test")
        mock_int.assert_called_once_with("count", default=5)
        mock_float.assert_called_once_with("ratio", default=0.5)
        mock_confirm.assert_called_once_with("enabled", default=True)

    @patch("rich.prompt.Prompt.ask")
    def test_prompt_with_none_response(self, mock_ask):
        """Test prompting when user response is None (should keep original value)."""
        schema = {"name": "default", "description": "default_desc"}
        mock_ask.side_effect = ["new_name", None]

        result = prompt_schema(schema)

        assert result == {"name": "new_name", "description": "default_desc"}  # Should keep original default

    @patch("rich.prompt.Prompt.ask")
    def test_prompt_keyboard_interrupt(self, mock_ask):
        """Test handling KeyboardInterrupt during prompting."""
        schema = {"name": "default"}
        mock_ask.side_effect = KeyboardInterrupt()

        result = prompt_schema(schema)

        assert result is None

    def test_prompt_unsupported_type(self):
        """Test error handling for unsupported data types."""
        schema = {"data": [1, 2, 3]}  # List is not supported

        with pytest.raises(NotImplementedError, match="Unsupported type: list"):
            prompt_schema(schema)

    def test_prompt_custom_object_type(self):
        """Test error handling for custom object types."""

        class CustomType:
            pass

        schema = {"custom": CustomType()}

        with pytest.raises(NotImplementedError, match="Unsupported type: CustomType"):
            prompt_schema(schema)

    @patch("rich.prompt.Prompt.ask")
    def test_empty_schema(self, mock_ask):
        """Test prompting with empty schema."""
        schema: dict[str, str] = {}

        result = prompt_schema(schema)

        assert result == {}
        mock_ask.assert_not_called()

    @patch("rich.prompt.Prompt.ask")
    def test_schema_original_not_modified(self, mock_ask):
        """Test that original schema is not modified during prompting."""
        original_schema = {"name": "original", "count": 42}
        schema_copy = original_schema.copy()
        mock_ask.side_effect = ["modified", 100]

        with patch("rich.prompt.IntPrompt.ask") as mock_int:
            mock_int.return_value = 100
            result = prompt_schema(original_schema)

        # Original schema should be unchanged
        assert original_schema == schema_copy
        # But result should have new values
        assert result == {"name": "modified", "count": 100}

    @patch("rich.prompt.Confirm.ask")
    def test_keyboard_interrupt_after_partial_input(self, mock_ask):
        """Test KeyboardInterrupt after partial input is provided."""
        schema = {"enabled": True, "debug": False}
        mock_ask.side_effect = [False, KeyboardInterrupt()]

        result = prompt_schema(schema)

        assert result is None
