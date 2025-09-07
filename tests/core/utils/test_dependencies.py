"""Tests for marimba.core.utils.dependencies module."""

import pytest
import typer

from marimba.core.utils.dependencies import (
    Platform,
    ToolDependency,
    ToolInfo,
    check_dependency_available,
    get_current_platform,
    get_tool_info,
    show_dependency_error,
    show_dependency_error_and_exit,
    validate_dependencies,
)


class TestPlatform:
    """Test Platform enum."""

    @pytest.mark.unit
    def test_platform_enum_values(self):
        """Test Platform enum has expected values."""
        assert Platform.WINDOWS.value == "windows"
        assert Platform.MACOS.value == "macos"
        assert Platform.LINUX.value == "linux"


class TestToolDependency:
    """Test ToolDependency enum."""

    @pytest.mark.unit
    def test_tool_dependency_enum_values(self):
        """Test ToolDependency enum has expected values."""
        assert ToolDependency.EXIFTOOL.value == "exiftool"
        assert ToolDependency.FFMPEG.value == "ffmpeg"
        assert ToolDependency.FFPROBE.value == "ffprobe"


class TestToolInfo:
    """Test ToolInfo dataclass."""

    @pytest.fixture
    def sample_tool_info(self):
        """Create sample ToolInfo for testing."""
        return ToolInfo(
            name="testtool",
            description="Test description",
            homepage="https://example.com",
            windows_instructions=["Windows install step 1", "Windows install step 2"],
            macos_instructions=["macOS install step 1"],
            linux_instructions=["Linux install step 1", "Linux install step 2", "Linux install step 3"],
        )

    @pytest.mark.unit
    def test_tool_info_initialization(self, sample_tool_info):
        """Test ToolInfo initialization."""
        assert sample_tool_info.name == "testtool"
        assert sample_tool_info.description == "Test description"
        assert sample_tool_info.homepage == "https://example.com"
        assert len(sample_tool_info.windows_instructions) == 2
        assert len(sample_tool_info.macos_instructions) == 1
        assert len(sample_tool_info.linux_instructions) == 3

    @pytest.mark.unit
    def test_get_platform_instructions_windows(self, sample_tool_info):
        """Test get_platform_instructions for Windows."""
        instructions = sample_tool_info.get_platform_instructions(Platform.WINDOWS)
        assert instructions == ["Windows install step 1", "Windows install step 2"]

    @pytest.mark.unit
    def test_get_platform_instructions_macos(self, sample_tool_info):
        """Test get_platform_instructions for macOS."""
        instructions = sample_tool_info.get_platform_instructions(Platform.MACOS)
        assert instructions == ["macOS install step 1"]

    @pytest.mark.unit
    def test_get_platform_instructions_linux(self, sample_tool_info):
        """Test get_platform_instructions for Linux."""
        instructions = sample_tool_info.get_platform_instructions(Platform.LINUX)
        assert instructions == ["Linux install step 1", "Linux install step 2", "Linux install step 3"]


class TestGetToolInfo:
    """Test get_tool_info function."""

    @pytest.mark.unit
    def test_get_tool_info_exiftool(self):
        """Test get_tool_info for exiftool."""
        tool_info = get_tool_info(ToolDependency.EXIFTOOL)
        assert tool_info.name == "exiftool"
        assert "ExifTool" in tool_info.description
        assert tool_info.homepage == "https://exiftool.org/"
        assert len(tool_info.windows_instructions) > 0
        assert len(tool_info.macos_instructions) > 0
        assert len(tool_info.linux_instructions) > 0

    @pytest.mark.unit
    def test_get_tool_info_ffmpeg(self):
        """Test get_tool_info for ffmpeg."""
        tool_info = get_tool_info(ToolDependency.FFMPEG)
        assert tool_info.name == "ffmpeg"
        assert "FFmpeg" in tool_info.description
        assert tool_info.homepage == "https://ffmpeg.org/"
        assert len(tool_info.windows_instructions) > 0
        assert len(tool_info.macos_instructions) > 0
        assert len(tool_info.linux_instructions) > 0


class TestGetCurrentPlatform:
    """Test get_current_platform function."""

    @pytest.mark.unit
    def test_get_current_platform_darwin(self, mocker):
        """Test get_current_platform returns macOS for Darwin."""
        mock_system = mocker.patch("platform.system")
        mock_system.return_value = "Darwin"
        assert get_current_platform() == Platform.MACOS

    @pytest.mark.unit
    def test_get_current_platform_windows(self, mocker):
        """Test get_current_platform returns Windows for Windows."""
        mock_system = mocker.patch("platform.system")
        mock_system.return_value = "Windows"
        assert get_current_platform() == Platform.WINDOWS

    @pytest.mark.unit
    def test_get_current_platform_linux(self, mocker):
        """Test get_current_platform returns Linux for Linux."""
        mock_system = mocker.patch("platform.system")
        mock_system.return_value = "Linux"
        assert get_current_platform() == Platform.LINUX

    @pytest.mark.unit
    def test_get_current_platform_unknown_defaults_to_linux(self, mocker):
        """Test get_current_platform defaults to Linux for unknown systems."""
        mock_system = mocker.patch("platform.system")
        mock_system.return_value = "UnknownOS"
        assert get_current_platform() == Platform.LINUX


class TestShowDependencyError:
    """Test show_dependency_error function."""

    @pytest.mark.unit
    def test_show_dependency_error_without_error_message(self, mocker):
        """Test show_dependency_error without error message."""
        mock_console_class = mocker.patch("marimba.core.utils.dependencies.Console")
        mock_get_platform = mocker.patch("marimba.core.utils.dependencies.get_current_platform")
        mock_console = mocker.Mock()
        mock_console_class.return_value = mock_console
        mock_get_platform.return_value = Platform.LINUX

        show_dependency_error(ToolDependency.EXIFTOOL)

        # Should call console.print() three times (empty line, panel, empty line)
        assert mock_console.print.call_count == 3

    @pytest.mark.unit
    def test_show_dependency_error_with_error_message(self, mocker):
        """Test show_dependency_error with error message."""
        mock_console_class = mocker.patch("marimba.core.utils.dependencies.Console")
        mock_get_platform = mocker.patch("marimba.core.utils.dependencies.get_current_platform")
        mock_console = mocker.Mock()
        mock_console_class.return_value = mock_console
        mock_get_platform.return_value = Platform.MACOS

        show_dependency_error(ToolDependency.FFMPEG, "Custom error message")

        # Should call console.print() three times
        assert mock_console.print.call_count == 3


class TestCheckDependencyAvailable:
    """Test check_dependency_available function."""

    @pytest.mark.unit
    def test_check_dependency_available_found(self, mocker):
        """Test check_dependency_available when tool is found."""
        mock_which = mocker.patch("shutil.which")
        mock_which.return_value = "/usr/bin/exiftool"

        result = check_dependency_available(ToolDependency.EXIFTOOL)

        assert result is True
        mock_which.assert_called_once_with("exiftool")

    @pytest.mark.unit
    def test_check_dependency_available_not_found(self, mocker):
        """Test check_dependency_available when tool is not found."""
        mock_which = mocker.patch("shutil.which")
        mock_which.return_value = None

        result = check_dependency_available(ToolDependency.EXIFTOOL)

        assert result is False
        mock_which.assert_called_once_with("exiftool")


class TestValidateDependencies:
    """Test validate_dependencies function."""

    @pytest.mark.unit
    def test_validate_dependencies_empty_list(self):
        """Test validate_dependencies with empty list."""
        # Should not raise any exception
        validate_dependencies([])

    @pytest.mark.unit
    def test_validate_dependencies_all_available(self, mocker):
        """Test validate_dependencies when all tools are available."""
        mock_check = mocker.patch("marimba.core.utils.dependencies.check_dependency_available")
        mock_check.return_value = True

        # Should not raise any exception
        validate_dependencies([ToolDependency.EXIFTOOL])

        mock_check.assert_called_once_with(ToolDependency.EXIFTOOL)

    @pytest.mark.unit
    def test_validate_dependencies_missing_tool(self, mocker):
        """Test validate_dependencies when a tool is missing."""
        mock_check = mocker.patch("marimba.core.utils.dependencies.check_dependency_available")
        mock_show_error_exit = mocker.patch("marimba.core.utils.dependencies.show_dependency_error_and_exit")
        mock_check.return_value = False

        validate_dependencies([ToolDependency.EXIFTOOL])

        mock_check.assert_called_once_with(ToolDependency.EXIFTOOL)
        mock_show_error_exit.assert_called_once_with(
            ToolDependency.EXIFTOOL,
            "Required dependency 'exiftool' is not available",
        )

    @pytest.mark.unit
    def test_validate_dependencies_ffmpeg_without_ffprobe(self, mocker):
        """Test validate_dependencies when ffmpeg is available but ffprobe is not."""
        mock_check = mocker.patch("marimba.core.utils.dependencies.check_dependency_available")
        mock_show_error_exit = mocker.patch("marimba.core.utils.dependencies.show_dependency_error_and_exit")

        def mock_check_side_effect(tool):
            if tool == ToolDependency.FFMPEG:
                return True
            return tool != ToolDependency.FFPROBE

        mock_check.side_effect = mock_check_side_effect

        validate_dependencies([ToolDependency.FFMPEG])

        # Should check both FFMPEG and FFPROBE
        assert mock_check.call_count == 2
        mock_show_error_exit.assert_called_once_with(ToolDependency.FFMPEG, "FFprobe (part of FFmpeg) is not available")

    @pytest.mark.unit
    def test_validate_dependencies_ffmpeg_with_ffprobe(self, mocker):
        """Test validate_dependencies when both ffmpeg and ffprobe are available."""
        mock_check = mocker.patch("marimba.core.utils.dependencies.check_dependency_available")
        mock_check.return_value = True

        # Should not raise any exception
        validate_dependencies([ToolDependency.FFMPEG])

        # Should check both FFMPEG and FFPROBE
        assert mock_check.call_count == 2


class TestShowDependencyErrorAndExit:
    """Test show_dependency_error_and_exit function."""

    @pytest.mark.unit
    def test_show_dependency_error_and_exit_default_exit_code(self, mocker):
        """Test show_dependency_error_and_exit with default exit code."""
        mock_console_class = mocker.patch("marimba.core.utils.dependencies.Console")
        mock_show_error = mocker.patch("marimba.core.utils.dependencies.show_dependency_error")
        mock_console = mocker.Mock()
        mock_console_class.return_value = mock_console

        with pytest.raises(typer.Exit) as exc_info:
            show_dependency_error_and_exit(ToolDependency.EXIFTOOL, "Test error")

        assert exc_info.value.exit_code == 1
        mock_show_error.assert_called_once_with(ToolDependency.EXIFTOOL, "Test error")
        assert mock_console.print.call_count == 2

    @pytest.mark.unit
    def test_show_dependency_error_and_exit_custom_exit_code(self, mocker):
        """Test show_dependency_error_and_exit with custom exit code."""
        mock_console_class = mocker.patch("marimba.core.utils.dependencies.Console")
        mock_show_error = mocker.patch("marimba.core.utils.dependencies.show_dependency_error")
        mock_console = mocker.Mock()
        mock_console_class.return_value = mock_console

        with pytest.raises(typer.Exit) as exc_info:
            show_dependency_error_and_exit(ToolDependency.FFMPEG, "Test error", exit_code=5)

        assert exc_info.value.exit_code == 5
        mock_show_error.assert_called_once_with(ToolDependency.FFMPEG, "Test error")
        assert mock_console.print.call_count == 2

    @pytest.mark.unit
    def test_show_dependency_error_and_exit_no_error_message(self, mocker):
        """Test show_dependency_error_and_exit without error message."""
        mock_console_class = mocker.patch("marimba.core.utils.dependencies.Console")
        mock_show_error = mocker.patch("marimba.core.utils.dependencies.show_dependency_error")
        mock_console = mocker.Mock()
        mock_console_class.return_value = mock_console

        with pytest.raises(typer.Exit) as exc_info:
            show_dependency_error_and_exit(ToolDependency.EXIFTOOL)

        assert exc_info.value.exit_code == 1
        mock_show_error.assert_called_once_with(ToolDependency.EXIFTOOL, "")
        assert mock_console.print.call_count == 2
