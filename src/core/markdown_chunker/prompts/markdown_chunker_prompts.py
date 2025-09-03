"""
Prompts for Markdown Chunker LLM processing
Contains all prompt templates used in the markdown chunking process
"""

# System message for LLM
SYSTEM_MESSAGE = "You are a professional Markdown text processing tool."

# Chunk markers for parsing LLM responses
CHUNK_START_MARKER = "===CHUNK START==="
CHUNK_END_MARKER = "===CHUNK END==="

# Table splitting prompt template
TABLE_SPLITTING_PROMPT_TEMPLATE = """You are a professional Markdown text processing tool.

⚠️ ALL requirements below are MANDATORY and must not be skipped for any reason.

Input data may include:
- Text descriptions
- Tables (Markdown format)
- Table captions

Tasks:
1. **Clean content**:
   - Keep all original characters unchanged.
   - Escape any \\ characters if needed to prevent processing errors.
   - Do NOT delete or modify any data.

2. **Split content into chunks**:
   - Each chunk MUST have token count **< {max_tokens}**.
   - If any chunk ≥ {max_tokens} tokens → continue splitting until requirement is met.
   - Estimation: 1 token ≈ 0.75 English words. {max_tokens} tokens ≈ {estimated_words} words.
   - Preserve table structure, **do NOT split in the middle of a row**.
   - If table exceeds limit → split into multiple smaller tables, each keeping the header.
   - If table has sub-headers → **ensure sub-header content is preserved if the entire group < {max_tokens} tokens**.
     If group > {max_tokens} tokens → **MUST split immediately**, even if it means separating sub-headers. When splitting, ensure headers are retained in split chunks.
   - Descriptions should be in separate chunks.
   - Captions must go with their corresponding tables.

   Example chunk with content: "Description + table + caption", table content:
   | Level | Content |
   | 1     | About 1200 tokens|
   | 2     | About 300 tokens|
   | 3     | About 100 tokens|
   OUTPUT: 
   ```
     {chunk_start_marker}
     <Description>
     {chunk_end_marker}
     {chunk_start_marker}
     | Level | Content |
     | 1     | About 600 tokens|
     {chunk_end_marker}
      {chunk_start_marker}
     | Level | Content |
     | 1     | About 600 tokens|
     {chunk_end_marker}
      {chunk_start_marker}
     | Level | Content |
     | 2     | About 300 tokens|
     | 3     | About 100 tokens|
     <caption>
     {chunk_end_marker}
     ```
   
3. **Verify before returning results**:
   - Review each chunk.
   - If any chunk ≥ {max_tokens} tokens → **go back to step 2 and split further**.
   - Only return results when **all chunks are < {max_tokens} tokens**.

4. **Return format**:
   - Each chunk between markers:
     ```
     {chunk_start_marker}
     <chunk content>
     {chunk_end_marker}
     ```
   - Do NOT add explanations or notes outside the chunks.
   - Do NOT use JSON format.
{retry_warning}

Content to process ({content_tokens} tokens):

{content}
"""

# Retry warning template for when splitting needs to be repeated
RETRY_WARNING_TEMPLATE = "\n⚠️ Warning: This chunk is still longer than {max_tokens} tokens and MUST be split into smaller pieces under {max_tokens} tokens."
