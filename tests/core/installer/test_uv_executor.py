import pytest

from marimba.core.installer.uv_executor import UvExecutor


@pytest.mark.integration
def test_uv_executor_output():
    uv_executor = UvExecutor.create()
    result = uv_executor("list")

    # Should succeed with empty list if no packages installed in current env
    assert result.error == ""


@pytest.mark.integration
def test_uv_executor_error():
    uv_executor = UvExecutor.create()

    with pytest.raises(UvExecutor.UvError):
        uv_executor("install abaöskjdsök")


@pytest.mark.unit
def test_uv_executor_create_uv_not_found(mocker):
    """Test UvExecutor.create() when uv is not found in PATH."""
    mock_which = mocker.patch("shutil.which")
    mock_which.return_value = None

    with pytest.raises(UvExecutor.UvError, match="uv executable not found in PATH"):
        UvExecutor.create()
