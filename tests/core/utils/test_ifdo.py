import tempfile
from collections.abc import Generator
from pathlib import Path
from uuid import uuid4

import pytest
from ifdo import iFDO
from ifdo.models import ImageSetHeader

from marimba.core.utils.ifdo import load_ifdo, save_ifdo


class TestIfdo:
    """
    Class to test the functionality of the iFDO class.
    """

    @pytest.fixture
    def ifdo_setup(self) -> Generator[tuple[Path, iFDO], None, None]:
        """Set up temporary directory and iFDO object for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ifdo_path = Path(temp_dir) / "test_ifdo.yaml"
            uuid = str(uuid4())
            ifdo = iFDO(
                image_set_header=ImageSetHeader(
                    image_set_name="test_image_set_name",
                    image_set_uuid=uuid,
                    image_set_handle="test_image_set_handle",
                ),
                image_set_items={},
            )
            yield ifdo_path, ifdo

    @pytest.mark.integration
    def test_load_ifdo(self, ifdo_setup: tuple[Path, iFDO]) -> None:
        ifdo_path, ifdo = ifdo_setup
        ifdo.save(ifdo_path)
        loaded_ifdo = load_ifdo(ifdo_path)
        assert ifdo == loaded_ifdo

    @pytest.mark.integration
    def test_save_ifdo(self, ifdo_setup: tuple[Path, iFDO]) -> None:
        ifdo_path, ifdo = ifdo_setup
        save_ifdo(ifdo, ifdo_path)
        assert ifdo_path.exists()
        loaded_ifdo = load_ifdo(ifdo_path)
        assert ifdo == loaded_ifdo
