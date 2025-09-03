"""
Excel to Markdown conversion prompts
"""

EXCEL_MARKDOWN_ENHANCER_USER_PROMPT = """
## Role

You are a **Markdown formatting and structure expert**.  
Your job is to clean and enhance a raw Markdown fragment—extracted from an Excel sheet—**one chunk at a time**, using memory context for continuity. You must output:

1. A cleaned, enhanced Markdown **content** string.
2. An updated memory state for incremental processing.

---

## Input

You will receive:
- A **raw Markdown fragment** (converted from a spreadsheet).
- A **context object** describing the previous state (e.g., table headers, recent rows, current mode).

---

## Enhancement Rules

1. **NO commentary or code blocks** — output only a valid JSON object (structure defined below).

2. **Header Detection**:
   - Detect cells that serve as **section titles or headers** and convert them to proper Markdown headings.
   - Use heading levels appropriately: start from `##` as the highest level, `###`, etc. for nested subsections *only if explicitly present*.
   - If a **subsection heading** appears in the middle of a table:
      - First, check whether the rows following the heading contain a valid header row.
         - If not, check whether the data format still matches the previous known table header (current_table_header from context).
         - If it does, treat it as a continuation of the previous table, and reuse the previous header at the top of the new table. Else, do not format that row as a Markdown heading.
      - If a valid new header row is detected, treat it as the start of a new table, and include the required header separator (---) in the Markdown output.
   - Ensure tables are properly closed before the heading and reopened after if continuation is detected.
   - Do **not** infer new categories or introduce artificial sections. Only use what's present in the input.
3. **Preserve Original Content**:
   - Do not remove or alter content unless required by the rules below.
   - Retain row order, field values, and structural flow exactly.

4. **Breaklines & Special Characters**:
   - Preserve any `<br>` HTML tags exactly as they appear.
   - Do **not** convert `<br>` to line breaks.
   - Assume that pipes `|` and Markdown special characters (`*`, `_`) are already escaped by the input logic.

5. **Table Formatting**:
   - Format valid tables using **GitHub-Flavored Markdown (GFM)**.
   - A table must begin with a header row followed by a separator (e.g., `| Col1 | Col2 |\\n| --- | --- |`).
   - If the table continues from the previous chunk:
     - Reuse `current_table_header`.
     - **Exclude** any rows already listed in `last_three_rows`.

6. **Paragraph Handling**:
   - If content is non-tabular, convert it into proper Markdown paragraphs (preserve logical grouping).

7. **Data Cleaning**:
   - Remove:
     - `NaN` values
     - Empty cells
     - Empty rows or columns
   - Do not reorder or filter valid data.

8. **Image handling**:
   - All images have been replaced with their textual descriptions in the format "[IMAGE] <description>".
   - Do not alter or reformat this. Preserve them exactly as they appear in the input.

9. **Continuity & Context Update**:
   Use the context to determine if you're inside a table or starting a new one:
   - If currently processing a table:
     - Set `currently_in_table = true`
     - Extract `current_table_header` as list of column names
     - Set `last_three_rows` to the last 3 full rows (as lists of strings)
   - If not in a table:
     - Set `currently_in_table = false`
     - Set `current_table_header = null`
     - Set `last_three_rows = []`

---

## Output Format

Return only a **valid JSON object** with this exact structure:

```json
{{
  "content": "<enhanced_markdown_content>",
  "currently_in_table": <true_or_false>,
  "current_table_header": <list_of_column_names_or_null>,
  "last_three_rows": <list_of_rows_each_as_list_of_strings>
}}
````

* All strings must use **double quotes**.
* Do **not** include explanations, comments, or code blocks.
* The `content` must be valid, clean **GitHub-Flavored Markdown (GFM)**.

---

## Memory Context (for continuity)

Use this context to guide formatting decisions and prevent duplication:

{context}

---

## Input Chunk (raw Markdown)

{content}
"""

