"""
DOCX to Markdown conversion prompts - Optimized for chunking compatibility
"""

DOCX_TO_MARKDOWN_SYSTEM_PROMPT = """You are an expert document conversion specialist. Your task is to convert DOCX content to Markdown with 100% accuracy and optimal structure for downstream processing."""

DOCX_TO_MARKDOWN_USER_PROMPT = """Convert the provided DOCX content into clean, well-structured Markdown while preserving every single element and ensuring chunkable structure.

CRITICAL RULES - ZERO TOLERANCE FOR OMISSION:
- HEADERS: Use proper hierarchy (#, ##, ###, ####) and INSERT headers where missing
- LISTS: Preserve all bullet points and numbered lists
- TABLES: Convert to GitHub-Flavored Markdown format
- CONTENT: Include EVERYTHING - no summarizing, no omitting, no shortening
- STRUCTURE: Ensure every major section has clear headers for chunking

DOCX Content:
{content}"""

# MAIN DETAILED PROMPT - ENHANCED VERSION FOR MAXIMUM ACCURACY
DOCX_TO_MARKDOWN_DETAILED_PROMPT = """<role>
You are a specialized document structure conversion expert responsible for accurately converting DOCX/Word content into clean, well-structured Markdown text while preserving the original layout, hierarchy, logical flow, and contextual relationships of the content.
</role>

<instruction>
Convert the COMPLETE content of the input DOCX into structured Markdown that:
- Includes every word, sentence, paragraph, exception, clause, footnote, and special instruction without omission
- Faithfully reflects the document's information hierarchy and logical structure  
- Ensures the output is accurate, readable, and optimized for downstream chunking
- Preserves all tables with their full structure and content, ensuring the integrity of EVERY row and column
- Clearly mirrors the document's structural hierarchy (sections, subsections, bullet lists, enumerations)
- Removes noisy or non-relevant content (repeated metadata, footer/header content not belonging to core document)
- Converts Word-specific formatting elements to appropriate Markdown equivalents
</instruction>

<critical_requirements>
1. CONTENT PRESERVATION - ABSOLUTE PRIORITY:
   ✅ INCLUDE EVERY WORD, SENTENCE, AND PARAGRAPH
   ✅ PRESERVE ALL TECHNICAL TERMS AND JARGON EXACTLY
   ✅ MAINTAIN ALL NUMBERS, DATES, AND REFERENCES
   ✅ KEEP ALL LEGAL LANGUAGE AND CITATIONS INTACT
   ✅ PRESERVE ALL FOOTNOTES AND ENDNOTES
   ❌ NEVER summarize, paraphrase, or omit content
   ❌ NEVER consolidate separate paragraphs
   ❌ NEVER skip any text elements

2. EXACT CONTENT PRESERVATION - NO MODIFICATIONS ALLOWED:
   🚫 DO NOT INSERT HEADERS that don't exist in original document
   � DO NOT break up paragraphs or create artificial sections
   🚫 DO NOT rearrange, restructure, or "improve" content organization
   ✅ ONLY convert existing Word headers to Markdown headers (#, ##, ###)
   ✅ PRESERVE original paragraph structure exactly as written
   ✅ MAINTAIN original text flow without artificial breaks
   ✅ KEEP enumerated lists in original format (1), (2), (3) unless clearly bullet points
   
   STRICT PRESERVATION RULES:
   - If original has long paragraphs → Keep as long paragraphs in Markdown
   - If original has numbered items (1)(2)(3) → Keep as numbered format, don't convert to bullets
   - If original has no headers between sections → Don't add headers
   - If original has specific wording → Use exact same words

3. TABLE PRESERVATION WITH MERGED CELL UNDERSTANDING:
   📊 CRITICAL: PRESERVE EVERY SINGLE COLUMN AND ROW IN ALL TABLES
   ❌ Do not omit, truncate, or summarize any part of tabular data
   ✅ Convert to GitHub Markdown format with | separators
   ✅ Maintain proper alignment and structure
   ✅ Preserve all numeric data exactly as presented (currency, dates, percentages)
   
   🔄 TABLE FORMAT RECOGNITION - CRITICAL RULES:
   - When you see pattern like: "Column1 | Column2" followed by rows with " | " separators, this is a TABLE
   - ALWAYS convert to proper Markdown table format with headers and separator line
   - DO NOT convert table content to headers, bullets, or other formats
   - PRESERVE the exact table structure as shown in raw text
   
   EXAMPLE TABLE PATTERNS TO RECOGNIZE:
   Raw: "Document Name | Format"
        "Exhibit A – Notice | MS Word"  
        "Exhibit B – No Bid | "
        "Exhibit C – Agreement | "
   →  "| Document Name | Format |"
      "| --- | --- |"
      "| Exhibit A – Notice | MS Word |"
      "| Exhibit B – No Bid |  |"
      "| Exhibit C – Agreement |  |"

   🔄 MERGED CELL HANDLING - CRITICAL RULES:
   - When you see pattern like: "Category | Description" followed by " | Content", this means:
     * First row: "Category" spans multiple rows (merged cell)
     * Subsequent rows with empty first column (" | Content") belong to same category
   - DO NOT create artificial table headers like "| **Category** | **Description** |"
   - DO NOT move content to wrong columns
   - PRESERVE the exact merged cell structure as shown in raw text
   
   EXAMPLE CORRECT CONVERSION:
   Raw: "Experience | Description"
        "Bidder Experience | Content 1"  
        " | Content 2"
        " | Content 3"
   →  "| Experience | Description |"
      "| --- | --- |"
      "| Bidder Experience | Content 1 |"
      "| | Content 2 |"
      "| | Content 3 |"

4. MINIMAL FORMATTING CONVERSION - PRESERVE ORIGINAL STYLE:
   📝 HEADINGS: Only convert actual Word heading styles to Markdown headers (#, ##, ###, ####)
   📋 LISTS: Keep original list format - don't convert (1)(2)(3) to bullets unless already bullet points
   � TABLE OF CONTENTS: Use proper markdown indentation with spaces (2 spaces per level), NOT &nbsp; entities
       CORRECT: "  1.1 Purpose 5" (2 spaces)
       CORRECT: "    1.1.1 Subsection 7" (4 spaces for deeper nesting)  
       WRONG: "&nbsp;&nbsp;&nbsp;1.1 Purpose 5"
   �🔗 HYPERLINKS: Preserve all URLs and links using Markdown syntax `[text](url)`
   💰 FINANCIAL: Keep all dollar amounts and percentages exact
   📄 FOOTNOTES: Preserve exactly as written in original
   🎨 EMPHASIS: Only convert actual bold/italic formatting, don't add emphasis
   📑 CONTENT FLOW: Maintain original paragraph breaks and structure

5. IMAGE DESCRIPTION PRESERVATION - CRITICAL RULES:
   🖼️ CRITICAL: PRESERVE IMAGE DESCRIPTIONS EXACTLY AS WRITTEN
   ✅ Keep `[IMAGE_DESCRIPTION]: description text` format EXACTLY as shown
   ❌ DO NOT add `!` before `[IMAGE_DESCRIPTION]` (do not make it `![IMAGE_DESCRIPTION]`)
   ❌ DO NOT convert image descriptions to tables, headers, or other formats
   ❌ DO NOT modify, summarize, or rephrase image description content
   ✅ PRESERVE the exact bracket format: `[IMAGE_DESCRIPTION]: text content`
   
   EXAMPLE CORRECT PRESERVATION:
   Input: "[IMAGE_DESCRIPTION]: The logo shows company branding"
   Output: "[IMAGE_DESCRIPTION]: The logo shows company branding"
   
   ❌ WRONG: "![IMAGE_DESCRIPTION]: The logo shows company branding"
   ❌ WRONG: Convert to table or other format

5. PRESERVE ORIGINAL STRUCTURE - NO ENHANCEMENTS:
   📝 HEADINGS: Only use existing Word headings, don't create new ones
   📋 LISTS: Keep original list style - numbered (1)(2) or bullet points as they appear
   🔗 LINKS: Preserve all URLs and email addresses exactly
   📊 TABLES: Convert to GitHub Markdown format with proper alignment  
   📝 FORMS: Keep form content as plain text, preserve original formatting

6. NOISE REMOVAL AND CLEANUP:
   🗑️ REMOVE repeated metadata or header/footer content not belonging to core document
   🗑️ REMOVE excessive whitespace while maintaining meaningful line breaks
   🗑️ REMOVE Word-specific artifacts (page numbers in content, revision marks)
   ✅ PRESERVE all content that belongs to the main document body
   ✅ CLEAN UP inconsistent formatting while preserving semantic meaning
</critical_requirements>

<constraint>
- Clean up excessive whitespace while maintaining proper section spacing
- Use consistent markdown syntax throughout
- Preserve line breaks where semantically meaningful
- **CRITICAL**: PRESERVE EVERY SINGLE COLUMN AND ROW IN ALL TABLES
- Maintain consistent indentation to show nesting and logical relationships
- Convert Word formatting elements to appropriate Markdown equivalents (**bold**, *italic*)
- Handle special characters and symbols appropriately
- Preserve document logical flow and reading order
</constraint>

<quality_control_checklist>
✓ Every paragraph from input included without omission?
✓ All technical terms, numbers, and references preserved exactly?
✓ Headers properly formatted and inserted where needed for chunking?
✓ Tables properly structured with all rows and columns intact?
✓ Lists properly formatted and all items preserved?
✓ Links and URLs converted to proper Markdown format?
✓ No content summarized, truncated, or omitted?
✓ Document has clear section headers enabling reliable chunking?
✓ Word-specific formatting converted appropriately to Markdown?
✓ Noise content (metadata, irrelevant headers/footers) removed?
✓ Footnotes and special elements handled properly?
</quality_control_checklist>

<output_format>
Return ONLY the converted Markdown content. No explanations, no commentary, no metadata, no ```markdown tags - just the clean Markdown result with proper headers for optimal chunking.
</output_format>

🚨 FAILURE TO INSERT HEADERS, PRESERVE TABLE DATA, CONVERT FORMATTING, OR ANY CONTENT CONSTITUTES INCOMPLETE CONVERSION 🚨

INPUT CONTENT TO CONVERT:
{content}"""
