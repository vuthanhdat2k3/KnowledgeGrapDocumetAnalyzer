from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseKGBuilder(ABC):
    """
    Simple abstract base class cho Knowledge Graph builders
    """
    @abstractmethod
    def build(self, *args, **kwargs):
        """
        Build the knowledge graph from the provided data.
        """
        pass
