"""
Centralized prompts for document conversion and processing
"""

from .pdf_prompts import (
    PDF_TO_MARKDOWN_SYSTEM_PROMPT,
    PDF_TO_MARKDOWN_USER_PROMPT,
    PDF_TO_MARKDOWN_DETAILED_PROMPT
)


__all__ = [
    # PDF prompts
    'PDF_TO_MARKDOWN_SYSTEM_PROMPT',
    'PDF_TO_MARKDOWN_USER_PROMPT', 
    'PDF_TO_MARKDOWN_DETAILED_PROMPT',
]
