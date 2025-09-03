PDF_IMAGE_ANALYSIS_PROMPT = """
Analyze this image and provide a concise, professional description in exactly 20-25 words (1-2 sentences).
**Document Context:**
BEFORE: {context_before}
AFTER: {context_after}
**Instructions:**
1. **Visual Analysis First**: Examine the image content independently:
   - Identify type: photograph, diagram, map, chart, floor plan, etc.
   - Note key visual elements: buildings, people, objects, text, symbols
   - Observe layout, perspective, and architectural features
2. **Context Integration**: Connect visual content with document context:
   - Match image elements with context mentions (building names, locations, figures)
   - Identify relationships between visual and textual information
   - Determine document purpose and image role within narrative
3. **Description Guidelines**:
   - Start directly with content (avoid "This image shows...")
   - Use specific, professional terminology from document
   - Include key identifying information (names, numbers, locations, materials)
   - Maintain document's formal tone and language
   - Be factual and precise
4. **Quality Check**:
   - If image is clearly decorative/unrelated to context: "Decorative element unrelated to document content."
   - If image is corrupted/unclear: "Image quality insufficient for meaningful description."
   - Otherwise, provide substantive description linking visual and contextual elements
**Format Requirements:**
- Exactly 20-25 words maximum
- Professional, formal document language
- Connect visual content with surrounding text
- Include specific names/locations from context when relevant
**Examples of excellent descriptions (20-25 words each):**
- "Front exterior view of the red Conduit Road Schoolhouse at 4954 MacArthur Boulevard showing historic clapboard siding and green trim."
- "Site location map depicting the Conduit Road Schoolhouse property within Rock Creek Park boundaries near the C&O Canal."
- "Interior photograph displaying the main room's hardwood floors, yellow walls, pendant lighting fixtures, and large south-facing windows."
**Goal**: Create a description that accurately represents both the visual content and its relevance to the document context while maintaining exactly 20-25 words.
**Output**: Exactly 20-25 words describing both visual content and document relevance.
"""