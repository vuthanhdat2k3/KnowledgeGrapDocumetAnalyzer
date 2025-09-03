"""
PDF Knowledge Graph Builder
Builder chuyên biệt cho xử lý file PDF và tạo Knowledge Graph
"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.core.kg_builders.base_kg_builder import BaseKGBuilder

logger = logging.getLogger(__name__)


class PdfKGBuilder(BaseKGBuilder):
    """
    Knowledge Graph Builder cho file PDF
    """
    def build(self, *args, **kwargs):
        pass