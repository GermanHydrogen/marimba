"""
Marimba Darwin Core Metadata Implementation.
"""

from marimba.core.schemas.base import BaseMetadata


class DarwinCoreMetadata(BaseMetadata):
    """
    Darwin Core metadata implementation for biological data interchange.

    This abstract class extends BaseMetadata to provide a specialized interface
    for Darwin Core metadata standard, commonly used in biodiversity and
    biological data management. Implementations should provide concrete
    behavior for all inherited abstract methods.
    """
