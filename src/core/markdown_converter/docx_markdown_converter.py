from src.core.markdown_converter.base_markdown_converter import BaseMarkdownConverter
from src.core.llm_client import LLMClientFactory
from src.core.markdown_converter.prompts.docx_prompts import DOCX_TO_MARKDOWN_DETAILED_PROMPT
from src.core.llm_client import LLMClientFactory
from src.core.markdown_converter.prompts.docx_prompts import DOCX_TO_MARKDOWN_DETAILED_PROMPT
import os
import logging
import re
import logging
import re
import subprocess
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DocxMarkdownConverter(BaseMarkdownConverter):
    def __init__(self, model_name: str = "gpt-4.1-nano", max_token: int = 16000):
        factory = LLMClientFactory()
        llm_info = factory.get_client(model_name)
        self.model = llm_info["model"]
        self.client = llm_info["client"]
        self.max_token = max_token

    def _normalize_filename(self, filename: str) -> str:
        """Normalize filename to avoid special characters but preserve Unicode."""
        # Only remove characters that are actually invalid in Windows filenames
        invalid_chars = r'[<>:"/\\|?*]'
        filename = re.sub(invalid_chars, "-", filename)
        # Remove extra whitespace
        filename = re.sub(r"\s+", " ", filename)
        return filename.strip()

    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file while preserving original document order."""
        try:
            doc = Document(file_path)
            text_content = []
            
            # Extract content in document order by iterating through document elements
            for element in doc.element.body:
                # Handle paragraphs
                if element.tag.endswith('p'):
                    # Find corresponding paragraph object
                    for para in doc.paragraphs:
                        if para._element == element:
                            if para.text.strip():
                                # Check if paragraph has heading style
                                heading_level = self._get_heading_level(para)
                                if heading_level:
                                    # Add appropriate markdown header
                                    markdown_header = "#" * heading_level + " " + para.text.strip()
                                    text_content.append(markdown_header)
                                else:
                                    text_content.append(para.text)
                            break
                
                # Handle tables
                elif element.tag.endswith('tbl'):
                    # Find corresponding table object
                    for table in doc.tables:
                        if table._element == element:
                            table_text = []
                            prev_row_cells = []
                            
                            for row_idx, row in enumerate(table.rows):
                                row_text = []
                                current_row_cells = []
                                
                                for cell_idx, cell in enumerate(row.cells):
                                    cell_text = cell.text.strip()
                                    current_row_cells.append(cell_text)
                                    
                                    # Check if this cell is the same as previous row (merged cell)
                                    if (row_idx > 0 and cell_idx < len(prev_row_cells) and 
                                        cell_text == prev_row_cells[cell_idx] and cell_text != ""):
                                        # This is likely a merged cell, use empty string for subsequent rows
                                        row_text.append("")
                                    else:
                                        row_text.append(cell_text)
                                
                                table_text.append(" | ".join(row_text))
                                prev_row_cells = current_row_cells
                            
                            text_content.append("\n".join(table_text))
                            break
            
            return "\n\n".join(text_content)
        except Exception as e:
            logging.error(f"Error extracting text from DOCX: {e}")
            raise

    def _get_heading_level(self, paragraph):
        """Determine heading level from paragraph style"""
        try:
            style_name = paragraph.style.name.lower()
            
            # Check for standard heading styles
            if 'heading 1' in style_name:
                return 1
            elif 'heading 2' in style_name:
                return 2
            elif 'heading 3' in style_name:
                return 3
            elif 'heading 4' in style_name:
                return 4
            elif 'heading 5' in style_name:
                return 5
            elif 'heading 6' in style_name:
                return 6
            
            # Check for Vietnamese heading styles (Title, Subtitle, etc.)
            if 'title' in style_name:
                return 1
            elif 'subtitle' in style_name:
                return 2
            
            # Check for outline levels
            if hasattr(paragraph, '_p') and hasattr(paragraph._p, 'pPr'):
                pPr = paragraph._p.pPr
                if pPr is not None:
                    from docx.oxml.ns import qn
                    outline_lvl = pPr.find(qn('w:outlineLvl'))
                    if outline_lvl is not None:
                        level = int(outline_lvl.get(qn('w:val')))
                        return min(level + 1, 6)  # Convert 0-based to 1-based, max 6
            
            return None
            
        except Exception as e:
            logging.debug(f"Could not determine heading level: {e}")
            return None

    def _split_text(self, text: str) -> list:
        """Split content into chunks not exceeding `max_token` characters."""
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= self.max_token:
                current += para + "\n\n"
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = para + "\n\n"
        
        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _build_prompt(self, content: str) -> str:
        """Build prompt for LLM."""
        return DOCX_TO_MARKDOWN_DETAILED_PROMPT.format(content=content)

    def convert_to_markdown(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        input_path = file_path

        # Handle .doc files by converting to .docx first
        if ext == ".doc":
            docx_path = str(Path(file_path).with_suffix('.docx'))
            try:
                # On Windows, LibreOffice executable is soffice
                subprocess.run([
                    "soffice",
                    "soffice",
                    "--headless",
                    "--convert-to", "docx",
                    "--outdir", str(Path(file_path).parent),
                    file_path
                ], check=True)
                input_path = docx_path
            except subprocess.CalledProcessError:
                raise RuntimeError("Failed to convert .doc to .docx using LibreOffice.")

        elif ext != ".docx":
            raise ValueError("Only .doc or .docx files are supported.")

        # Extract text from DOCX
        logging.info(f"Reading DOCX file: {input_path}")
        full_text = self._extract_text_from_docx(input_path)
        
        if not full_text.strip():
            raise ValueError("No text content found in the DOCX file.")

        # Split text into chunks for LLM processing
        text_chunks = self._split_text(full_text)
        logging.info(f"Document split into {len(text_chunks)} chunk(s)")

        # Process each chunk with LLM
        all_md_parts = []
        for idx, chunk in enumerate(tqdm(text_chunks, desc="Converting chunks with LLM")):
            prompt = self._build_prompt(chunk)

            logging.info(f"Sending chunk {idx + 1} to LLM...")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert document conversion specialist."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=min(self.max_token, 16000),  # Giới hạn max_tokens để tương thích với hầu hết các model
                )

                markdown_text = response.choices[0].message.content
                all_md_parts.append(markdown_text)
                logging.info(f"Chunk {idx + 1} processed successfully.")
            except Exception as e:
                logging.error(f"Error processing chunk {idx + 1}: {e}")
                raise

        # Combine markdown parts together
        final_markdown = "\n\n".join(all_md_parts)

        # Save results
        safe_name = self._normalize_filename(os.path.basename(input_path))
        md_path = os.path.join(os.path.dirname(input_path), 
                               os.path.splitext(safe_name)[0] + "_llm_converted.md")
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_markdown)
        
        logging.info(f"Markdown written to: {md_path}")

        return md_path
    
if __name__ == "__main__":

    converter = DocxMarkdownConverter()
    file_path = "data/sample_documents/elpasoamiandmdmsrfpv19_10012019-final_described.docx"
    converter.convert_to_markdown(file_path)