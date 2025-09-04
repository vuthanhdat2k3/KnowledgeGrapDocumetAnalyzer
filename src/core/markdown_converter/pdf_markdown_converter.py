from src.core.markdown_converter.base_markdown_converter import BaseMarkdownConverter
from src.core.llm_client import LLMClientFactory
from src.core.markdown_converter.prompts.pdf_prompts import PDF_TO_MARKDOWN_DETAILED_PROMPT
import os
import json
import time
import fitz
import re
import logging
from tqdm import tqdm 

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PdfMarkdownConverter(BaseMarkdownConverter):
    def __init__(self, model_name:str="gemini-2.5-flash"):
        factory = LLMClientFactory()
        gpt_info = factory.get_client(model_name)
        self.model = gpt_info["model"]
        self.client = gpt_info["client"]

    def _normalize_filename(self, filename: str) -> str:
        filename = re.sub(r"[^a-zA-Z0-9\s\-\(\)\[\]]", "-", filename)
        filename = re.sub(r"\s+", " ", filename)
        return filename.strip()

    def _build_prompt(self, content: str) -> str:
        return PDF_TO_MARKDOWN_DETAILED_PROMPT.format(content=content)

    def _split_text(self, text: str, max_length: int = 10000) -> list:
        """Split content into chunks that don't exceed `max_length` characters."""
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= max_length:
                current += para + "\n\n"
            else:
                chunks.append(current.strip())
                current = para + "\n\n"
        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _detect_header_level(self, text_content, font_size=12, is_bold=False):
        """
        Detect header level based on content pattern and font characteristics
        """
        text_content = text_content.strip()
        
        # Skip if text is too long (likely not a header)
        if len(text_content) > 150:
            return 0
            
        # Skip page headers and false positives 
        if re.match(r'^RFP:\s*Banking\s*Services\s*$', text_content, re.IGNORECASE):
            return 0  # Don't treat page headers as structural headers
        if re.match(r'^\d+$', text_content):  # Just numbers
            return 0
            
        # H1: Main document sections
        if re.match(r'^REQUEST FOR PROPOSAL', text_content, re.IGNORECASE):
            return 1
        if re.match(r'^SECTION\s+[IVX\d]+', text_content, re.IGNORECASE):
            return 1
            
        # H2: Major subsections (A., B., C.)
        if re.match(r'^[A-Z]\.?\s+[A-Z]', text_content):
            return 2
        if re.match(r'^(Introduction|Questions|Submission|Award|Note)$', text_content):
            return 2
            
        # H3: Numbered subsections (1., 2., 3.)  
        if re.match(r'^\d+\.?\s+[A-Z]', text_content):
            return 3
            
        # H4: Minor subsections based on font
        if font_size >= 13 and is_bold and len(text_content) < 80:
            return 4
            
        return 0

    def _detect_tables_and_format_text(self, page):
        """
        Detect tables and return both table info and formatted text blocks
        """
        try:
            tables = page.find_tables()
            table_regions = []
            
            for table_index, table in enumerate(tables):
                bbox = table.bbox
                
                # Extract actual table data for debugging
                table_data = table.extract()
                if table_data:
                    logging.info(f"Table {table_index} data preview:")
                    for i, row in enumerate(table_data[:3]):  # Show first 3 rows
                        logging.info(f"  Row {i}: {row}")
                        if i >= 2:  # Only show first 3 rows
                            if len(table_data) > 3:
                                logging.info(f"  ... and {len(table_data) - 3} more rows")
                            break
                
                table_regions.append({
                    'bbox': bbox,
                    'y_center': (bbox[1] + bbox[3]) / 2,
                    'index': table_index,
                    'data': table_data
                })
                logging.info(f"Detected table {table_index} at position {(bbox[1] + bbox[3]) / 2}")
            
            return table_regions
        except Exception as e:
            logging.warning(f"Table detection failed: {e}")
            return []

    def _convert_table_data_to_markdown(self, table_data):
        """
        Convert raw table data to clean markdown format
        """
        if not table_data or len(table_data) == 0:
            return ""
        
        # Filter out completely empty rows
        filtered_rows = []
        for row in table_data:
            if any(cell and str(cell).strip() for cell in row):  # At least one non-empty cell
                filtered_rows.append(row)
        
        if not filtered_rows:
            return ""
        
        markdown_lines = []
        
        # Process each row
        for row_idx, row in enumerate(filtered_rows):
            # Clean and format cells
            formatted_cells = []
            for cell in row:
                if cell is None:
                    formatted_cells.append("")
                else:
                    # Clean cell content - handle newlines in headers
                    cell_content = str(cell).strip().replace('\n', ' ')
                    # Replace | with &#124; to avoid markdown table issues
                    cell_content = cell_content.replace("|", "&#124;")
                    formatted_cells.append(cell_content)
            
            # Create markdown table row
            markdown_row = "| " + " | ".join(formatted_cells) + " |"
            markdown_lines.append(markdown_row)
            
            # Add header separator after first row (if it looks like a header)
            if row_idx == 0:
                # Create separator row
                separator_cells = ["---"] * len(formatted_cells)
                separator_row = "| " + " | ".join(separator_cells) + " |"
                markdown_lines.append(separator_row)
        
        return "\n".join(markdown_lines)

    def extract_with_layout_analysis(self, pdf_path):
        """
        Enhanced layout analysis with header detection and table formatting
        """
        doc = fitz.open(pdf_path)
        result = ""
        global_index = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Detect table regions for context
            table_regions = self._detect_tables_and_format_text(page)
            
            # Get detailed text info for font analysis
            text_dict = page.get_text("dict")
            
            # Use get_text("blocks") to get block information
            blocks = page.get_text("blocks")
            
            # Get detailed image information - SAME LOGIC AS PdfImageDescriber
            images_info = []
            
            # Method 1: Standard get_images()
            for img_index, img in enumerate(page.get_images()):
                try:
                    # Get image bbox
                    img_rects = page.get_image_rects(img[0])
                    for rect in img_rects:
                        images_info.append({
                            'bbox': rect,
                            'y_center': (rect[1] + rect[3]) / 2,
                            'source': 'get_images'
                        })
                except Exception as e:
                    logging.warning(f"Could not get rects for image {img_index} on page {page_num}: {e}")
                    continue
            
            # Method 2: Extract from text blocks (for embedded/replaced images)
            image_block_count = 0
            
            for block in text_dict.get("blocks", []):
                if block.get("type") == 1:  # Image block
                    image_block_count += 1
                    try:
                        bbox = block.get("bbox")
                        if bbox and "image" in block:
                            # Check if it's a duplicate (same logic as PdfImageDescriber)
                            is_duplicate = False
                            for existing in images_info:
                                if (abs(existing['y_center'] - (bbox[1] + bbox[3]) / 2) < 20):
                                    is_duplicate = True
                                    break
                            
                            # Only add if not duplicate and reasonable size
                            if not is_duplicate and (bbox[2] - bbox[0]) > 20 and (bbox[3] - bbox[1]) > 20:
                                rect = fitz.Rect(bbox)
                                images_info.append({
                                    'bbox': rect,
                                    'y_center': (rect[1] + rect[3]) / 2,
                                    'source': 'text_block'
                                })
                                logging.info(f"Found image in text block {image_block_count} on page {page_num}")
                    
                    except Exception as e:
                        logging.warning(f"Error processing text block {image_block_count} on page {page_num}: {e}")
                        continue
            
            # Create list of all elements
            all_elements = []
            
            # Add text blocks with header detection and table formatting
            for block in blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                if block_type == 0:  # Text block
                    text_content = text.strip()
                    if not text_content:
                        continue
                    
                    # Get font info for this block
                    font_size = 12
                    is_bold = False
                    
                    # Find matching block in detailed text info
                    for detail_block in text_dict.get("blocks", []):
                        if detail_block.get("type") == 0:
                            detail_bbox = detail_block.get("bbox", [])
                            if len(detail_bbox) >= 4 and abs(detail_bbox[1] - y0) < 5:
                                # Get font info from first span
                                for line in detail_block.get("lines", []):
                                    for span in line.get("spans", []):
                                        font_size = span.get("size", 12)
                                        font_name = span.get("font", "").lower()
                                        is_bold = "bold" in font_name or span.get("flags", 0) & 16 > 0
                                        break
                                    break
                                break
                    
                    # Handle multi-line text blocks that might contain headers
                    lines = text_content.split('\n')
                    processed_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            processed_lines.append('')
                            continue
                            
                        # Detect header level for each line
                        header_level = self._detect_header_level(line, font_size, is_bold)
                        

                        
                        # Format as header if detected
                        if header_level > 0:
                            formatted_line = "#" * header_level + " " + line
                            logging.debug(f"Detected H{header_level}: {line[:50]}...")
                            processed_lines.append(formatted_line)
                        else:
                            processed_lines.append(line)
                    
                    formatted_content = '\n'.join(processed_lines)
                    
                    # Check if in table region and skip individual text blocks in tables
                    in_table = False
                    for table in table_regions:
                        table_bbox = table['bbox']
                        if (x0 >= table_bbox[0] - 10 and x1 <= table_bbox[2] + 10 and
                            y0 >= table_bbox[1] - 10 and y1 <= table_bbox[3] + 10):
                            # Log table text for debugging
                            logging.info(f"📊 Table text block (SKIPPED): '{text_content[:100]}...'")
                            in_table = True
                            break
                    
                    # Skip text blocks that are part of tables - we'll add table markdown separately
                    if in_table:
                        continue
                    
                    all_elements.append({
                        'type': 'text',
                        'y': y0,
                        'content': formatted_content
                    })
            
            # Add actual table content using extracted data
            for table in table_regions:
                if 'data' in table and table['data']:
                    table_markdown = self._convert_table_data_to_markdown(table['data'])
                    if table_markdown:
                        all_elements.append({
                            'type': 'table',
                            'y': table['y_center'],
                            'content': f"[TABLE_DATA]\n{table_markdown}"
                        })
                        logging.info(f"📊 Added table markdown at position {table['y_center']}")

            # Add image placeholders with same order as PdfImageDescriber
            for img_info in images_info:
                all_elements.append({
                    'type': 'image',
                    'y': img_info['y_center'],
                    'content': f'[image placeholder {global_index}]',
                    'source': img_info.get('source', 'unknown')
                })
                global_index += 1
            
            # Sort by Y position
            all_elements.sort(key=lambda x: x['y'])
            
            # Combine content
            page_content = ""
            for element in all_elements:
                page_content += element['content'] + "\n\n"
            
            result += page_content
        
        doc.close()
        return result

    def _load_image_descriptions(self, pdf_path: str) -> dict:
        """Load image descriptions from JSON file"""
        json_file = self._get_description_file_path(pdf_path)
        
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                descriptions = json.load(f)
            logging.info(f"Loaded {len(descriptions)} image descriptions from {json_file}")
            return descriptions
        else:
            logging.warning(f"Image descriptions file not found: {json_file}")
            return {}

    def _replace_placeholders_with_descriptions(self, text: str, descriptions: dict) -> str:
        """Replace image placeholders with actual descriptions, ensuring proper positioning"""
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Find placeholder in this line
            placeholder_found = False
            for index, desc_info in descriptions.items():
                placeholder = f'[image placeholder {index}]'
                if placeholder in line:
                    description = desc_info['description']
                    
                    # If placeholder is in the middle of a line with other text
                    if line.strip() != placeholder:
                        # Split text before and after placeholder
                        parts = line.split(placeholder)
                        if len(parts) == 2:
                            text_before = parts[0].strip()
                            text_after = parts[1].strip()
                            
                            # Combine text before and after (if any)
                            combined_text = text_before
                            if text_after:
                                combined_text += " " + text_after if combined_text else text_after
                            
                            # Add text to current line
                            if combined_text:
                                processed_lines.append(combined_text)
                            
                            # Add description as a separate paragraph
                            processed_lines.append(f'[IMAGE_DESCRIPTION]: {description}')
                            placeholder_found = True
                            break
                    else:
                        # Placeholder occupies the entire line
                        processed_lines.append(f'[IMAGE_DESCRIPTION]: {description}')
                        placeholder_found = True
                        break
            
            # If no placeholder found in this line
            if not placeholder_found:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines)

    def _check_and_restore_missing_descriptions(self, markdown_content: str, original_descriptions: dict) -> str:
        """
        Check and restore missing descriptions after conversion
        """
        if not original_descriptions:
            return markdown_content

        missing_descriptions = []
        preserved_count = 0

        # Check each description
        for index, desc_info in original_descriptions.items():
            description = desc_info['description']
            image_tag = f"[IMAGE_DESCRIPTION]: {description}"
            if image_tag in markdown_content:
                preserved_count += 1
                logging.info(f"✓ Description {index} preserved: {description[:30]}...")
            else:
                missing_descriptions.append({
                    'index': index,
                    'description': description,
                    'context_before': desc_info.get('context_before', ''),
                    'context_after': desc_info.get('context_after', '')
                })
                logging.warning(f"✗ Description {index} MISSING: {description[:30]}...")

        logging.info(f"Image preservation: {preserved_count}/{len(original_descriptions)} descriptions preserved")

        if not missing_descriptions:
            logging.info("🎉 All image descriptions preserved successfully!")
            return markdown_content

        # Restore missing descriptions
        restored_content = markdown_content

        for missing in missing_descriptions:
            description = missing['description']
            context_before = missing['context_before'].strip()
            context_after = missing['context_after'].strip()

            logging.info(f"🔧 Restoring missing description: {description[:50]}...")

            # Find best position to insert description
            insertion_position = self._find_insertion_position(
                restored_content, context_before, context_after
            )

            description_text = f'\n\n[IMAGE_DESCRIPTION]: {description}\n\n'
            if insertion_position != -1:
                # Insert description at found position
                restored_content = (
                    restored_content[:insertion_position] + 
                    description_text + 
                    restored_content[insertion_position:]
                )
                logging.info(f"✓ Restored description at position {insertion_position}")
            else:
                # If no position found, add after first section
                lines = restored_content.split('\n')
                # Find position after first header
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('#') and i < len(lines) - 1:
                        insert_index = i + 1
                        break
                lines.insert(insert_index, f'[IMAGE_DESCRIPTION]: {description}')
                lines.insert(insert_index + 1, '')  # Add empty line
                restored_content = '\n'.join(lines)
                logging.info(f"✓ Appended description after first header")

        if missing_descriptions:
            logging.info(f"🔧 Restoration completed. Added {len(missing_descriptions)} missing descriptions.")

        return restored_content

    def _find_insertion_position(self, content: str, context_before: str, context_after: str) -> int:
        """
        Find best position to insert description based on context
        """
        # Try finding by context_after first
        if context_after and len(context_after) > 15:
            # Get keywords from context_after
            context_words = context_after.split()[:5]  # Take first 5 words
            for i in range(len(context_words), 0, -1):
                search_phrase = ' '.join(context_words[:i])
                if search_phrase in content:
                    position = content.find(search_phrase)
                    logging.debug(f"Found insertion position by context_after: {position}")
                    return position
        
        # Try finding by context_before
        if context_before and len(context_before) > 15:
            # Get keywords from context_before  
            context_words = context_before.split()[-5:]  # Take last 5 words
            for i in range(len(context_words), 0, -1):
                search_phrase = ' '.join(context_words[-i:])
                if search_phrase in content:
                    position = content.find(search_phrase) + len(search_phrase)
                    logging.debug(f"Found insertion position by context_before: {position}")
                    return position
        
        return -1

    def _post_process_markdown(self, markdown_content: str) -> str:
        """
        Post-process markdown to fix common LLM conversion issues
        """
        logging.info("🔧 Post-processing markdown to fix common issues...")
        
        # 1. Fix IMAGE_DESCRIPTION format - remove ! prefix added by LLM
        markdown_content = re.sub(
            r'!\[IMAGE_DESCRIPTION\]:',  # Match ![IMAGE_DESCRIPTION]:
            r'[IMAGE_DESCRIPTION]:',      # Replace with [IMAGE_DESCRIPTION]:
            markdown_content
        )
        
        # 2. Remove page markers that LLM might have missed
        page_markers = [
            r'# Page \| \d+',           # "# Page | 1"
            r'Page \| \d+',             # "Page | 1"
            r'===+ PAGE \d+ ===+',      # "===== PAGE 1 ====="
            r'--- PAGE \d+ ---',        # "--- PAGE 1 ---"
            r'Page \d+',                # "Page 1"
        ]
        
        for pattern in page_markers:
            markdown_content = re.sub(pattern, '', markdown_content, flags=re.IGNORECASE)
        
        # 3. Clean up excessive empty lines (more than 2 consecutive)
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
        
        # 4. Fix malformed header markers
        # Fix "# #" -> "##" (two separate # with space)
        markdown_content = re.sub(r'^#\s+#\s+', '## ', markdown_content, flags=re.MULTILINE)
        # Fix "# # #" -> "###" (three separate # with spaces)
        markdown_content = re.sub(r'^#\s+#\s+#\s+', '### ', markdown_content, flags=re.MULTILINE)
        # Fix "# # ##" -> "###" (mixed patterns)
        markdown_content = re.sub(r'^#\s+#\s+##\s+', '### ', markdown_content, flags=re.MULTILINE)
        # Fix any other duplicate header patterns
        markdown_content = re.sub(r'^(#+)\s+\1\s+', r'\1\1 ', markdown_content, flags=re.MULTILINE)
        
        # 5. Ensure proper spacing around IMAGE_DESCRIPTION
        markdown_content = re.sub(
            r'\n(\[IMAGE_DESCRIPTION\]:)',
            r'\n\n\1',
            markdown_content
        )
        
        logging.info("✓ Post-processing completed")
        return markdown_content.strip()
    
    def _extract_response_text(self, response, idx: int) -> str:
        """Trích xuất văn bản từ response của Gemini, fallback khi rỗng."""
        texts = []
        # Chỉ xử lý Gemini candidates
        if hasattr(response, "candidates"):
            for cand in getattr(response, "candidates", []):
                if getattr(cand, "content", None):
                    for part in getattr(cand.content, "parts", []):
                        if hasattr(part, "text") and part.text:
                            texts.append(part.text)
        # Fallback sang response.text (nếu Gemini trả về dạng này)
        if not texts and hasattr(response, "text"):
            if response.text:
                texts.append(response.text)
        if not texts:
            logging.warning(f"Chunk {idx+1} trả về rỗng → dùng placeholder")
            return f"[Không thể tạo nội dung cho phần {idx+1}]"
        return "\n".join(texts)
    
    def convert_to_markdown(self, file_path: str) -> str:
        logging.info(f"Reading PDF file: {file_path}")
        
        # Extract text with placeholders
        full_text = self.extract_with_layout_analysis(file_path)
        
        # Load image descriptions and replace placeholders
        descriptions = self._load_image_descriptions(file_path)
        if descriptions:
            full_text = self._replace_placeholders_with_descriptions(full_text, descriptions)
            logging.info(f"Replaced {len(descriptions)} image placeholders with descriptions")
        
        # Split into chunks for LLM processing
        text_chunks = self._split_text(full_text)
        logging.info(f"Document split into {len(text_chunks)} chunk(s)")
        
        # Process each chunk with LLM
        all_md_parts = []
        for idx, chunk in enumerate(tqdm(text_chunks, desc="Converting chunks with GPT")):
            prompt = self._build_prompt(chunk)

            logging.info(f"Sending chunk {idx + 1} to LLM...")
            # Retry với backoff khi gặp 503
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[
                            {"role": "user", "parts":[{"text": f"You are a helpful assistant.\n\n{prompt}"}]}
                        ],
                    )
                    markdown_text = self._extract_response_text(response, idx)
                    all_md_parts.append(markdown_text)
                    logging.info(f"Chunk {idx+1} processed successfully.")
                    break
                except Exception as e:
                    if "503" in str(e) and attempt < 2:
                        wait_time = 2 ** attempt
                        logging.warning(f"Chunk {idx+1} bị 503, thử lại sau {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    logging.error(f"Error processing chunk {idx+1}: {e}")
                    all_md_parts.append(f"[Lỗi khi xử lý chunk {idx+1}]")
                    break
            
        # Combine all parts
        final_markdown = "\n\n".join(all_md_parts)

        # POST-PROCESSING: Fix common LLM issues
        final_markdown = self._post_process_markdown(final_markdown)

        # CHECK AND RESTORE MISSING DESCRIPTIONS
        if descriptions:
            logging.info("🔍 Checking for missing image descriptions...")
            final_markdown = self._check_and_restore_missing_descriptions(final_markdown, descriptions)

        # Save to file
        md_suffix = "_with_descriptions" if descriptions else ""
        md_path = os.path.splitext(file_path)[0] + f"{md_suffix}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_markdown)

        logging.info(f"Markdown written to: {md_path}")
        
        # Clean up: Delete description JSON file after successful conversion
        if descriptions:
            json_file = self._get_description_file_path(file_path)
            if os.path.exists(json_file):
                try:
                    os.remove(json_file)
                    logging.info(f"🗑️ Cleaned up description file: {json_file}")
                except Exception as e:
                    logging.warning(f"Could not delete description file {json_file}: {e}")
        
        return md_path

    def _get_description_file_path(self, pdf_path: str) -> str:
        """Get the path to the description JSON file for a given PDF"""
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        json_filename = f"{base_name}_image_descriptions_vision.json"
        return os.path.join("src", "core", "image_describer", "output", "pdf_describer", json_filename)

if __name__ == "__main__":
    converter = PdfMarkdownConverter()
    
    # Test with sample PDF file  
    test_file = "data/sample_documents/project_doc_1_image.pdf"
    
    try:
        result_path = converter.convert_to_markdown(test_file)
        print(f"Conversion completed successfully!")
        print(f"Output file: {result_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")
