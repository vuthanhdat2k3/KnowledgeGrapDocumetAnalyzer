"""
PDF to Markdown conversion prompts - REVISED VERSION FOR MAXIMUM ACCURACY
"""

PDF_TO_MARKDOWN_SYSTEM_PROMPT = """You are an expert document conversion specialist. Your task is to convert PDF content to Markdown with 100% accuracy and completeness."""

PDF_TO_MARKDOWN_USER_PROMPT = """Convert the provided PDF content into clean, well-structured Markdown while preserving every single element.

CRITICAL RULES - ZERO TOLERANCE FOR OMISSION:
- PAGE MARKERS: Remove page separators (e.g., "==== PAGE 1 ====", "--- PAGE 2 ---") 
- HEADERS: Use proper hierarchy (#, ##, ###, ####)
- LISTS: Preserve all bullet points and numbered lists
- TABLES: Convert to GitHub-Flavored Markdown format
- IMAGES: Convert [description] markers to [IMAGE_DESCRIPTION]: description format
- CONTENT: Include EVERYTHING - no summarizing, no omitting, no shortening

PDF Content:
{content}"""

# CONCISE BUT COMPREHENSIVE PROMPT FOR MAXIMUM ACCURACY
PDF_TO_MARKDOWN_DETAILED_PROMPT = """Convert PDF content to structured Markdown following these rules:

CRITICAL REQUIREMENTS:
1. PRESERVE EVERYTHING: Include every word, sentence, paragraph - NO omission or summarization
2. STRUCTURE: Use proper header hierarchy based on document structure:
   - Main sections → # (H1)
   - Subsections → ## (H2) 
   - Minor sections → ### (H3)
   - Keep original document structure and hierarchy
3. TABLES: 
   - Keep [TABLE_DATA] sections as-is - they are already properly formatted markdown tables
   - DO NOT convert bullet lists or simple lists to tables
   - Only convert actual tabular data (rows/columns with headers) to markdown tables
4. LISTS: Preserve bullet points (•) and sub-items (o) as markdown lists - DO NOT convert to tables:
   - • Main item → - Main item
   - o Sub item → - Sub item (indented)
5. IMAGES: Convert `[image placeholder X]` to `[IMAGE_DESCRIPTION]: description_content` (NO ! prefix)
6. CLEAN UP: Remove page markers only (===PAGE 1===, ---PAGE 2---)
EXAMPLES:
- `[image placeholder 5]` → `[IMAGE_DESCRIPTION]: description_content`
- `[TABLE_DATA]` sections → Keep the markdown table exactly as provided
- `• Main item` → `- Main item` (bullet lists stay as lists, NOT tables)
- `o Sub item` → `  - Sub item` (indented sub-items)
- SECTION I → # SECTION I, A. Financial → ## A. Financial

Convert the following PDF content:

{content}"""

