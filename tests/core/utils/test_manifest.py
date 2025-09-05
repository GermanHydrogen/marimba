from pathlib import Path

import pytest

from marimba.core.utils.manifest import Manifest


@pytest.mark.unit
def test_get_subdirectories() -> None:
    base_dir = Path("tmp")
    files = {base_dir / "data" / "event" / "image.jpg", base_dir / "data" / "event" / "another.jpg"}
    sub_directories = Manifest._get_sub_directories(files, base_dir)

    assert sub_directories == {base_dir / "data", base_dir / "data" / "event"}
