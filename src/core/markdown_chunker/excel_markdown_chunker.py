import re
import json
import logging
from typing import List, Dict, Any, Tuple

import tiktoken

from src.core.markdown_chunker.base_markdown_chunker import BaseMarkdownChunker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class ExcelMarkdownChunker(BaseMarkdownChunker):
    def __init__(self, token_limit: int = 650, model: str = "gpt-4o"):
        """
        Initialize the chunker with token limits and settings.
        
        Args:
            token_limit: Maximum tokens per chunk
            model: Model name for token counting
        """
        self.token_limit = token_limit
        self.encoding = tiktoken.encoding_for_model(model)
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def chunk_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Chunk a markdown file into smaller parts following the specified pipeline.
        Returns a list of chunks in JSON format.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        doc = Document(page_content=content, metadata={'source': file_path})
        chunks = self._chunk_document(doc)
        
        return self._chunks_to_json(chunks)
    
    def _chunk_document(self, doc: Document) -> List[Document]:
        """Main chunking pipeline following the specified steps."""
        # Step 1: Check if whole document fits
        if self.count_tokens(doc.page_content) <= self.token_limit:
            return [doc]
        
        # Step 2: Split by headers
        header_chunks = self._split_by_headers(doc)
        
        # Process each header chunk
        final_chunks = []
        for chunk in header_chunks:
            if self.count_tokens(chunk.page_content) <= self.token_limit:
                final_chunks.append(chunk)
            else:
                # Further split oversized chunks
                sub_chunks = self._process_oversized_chunk(chunk)
                final_chunks.extend(sub_chunks)
        
        return final_chunks
    
    def _split_by_headers(self, doc: Document) -> List[Document]:
        """Recursively split by headers, diving deeper only if a section is too large."""
        root = self._parse_header_structure(doc)
        return self._recursive_chunk_section(root)
    
    def _get_header_prefix(self, section: Dict) -> str:
        header_prefix = []
        for level in range(1, section["level"] + 1):
            header_text = section['header_path'].get(f'Header {level}')
            if header_text:
                header_prefix.append(f'{"#" * level} {header_text}')
        
        return '\n\n'.join(header_prefix)
        

    def _recursive_chunk_section(self, section: Dict, current_level: int = 0) -> List[Document]:
        chunks = []

        # If we've reached the maximum level or there are no children, create a chunk for the entire section
        if current_level > 6 or not section.get('children'):
            content = self._section_to_content(section)
            if content.strip():
                chunks.append(Document(
                    page_content=content.strip(),
                    metadata=section['metadata']
                ))
            return chunks

        # Handle content between headers (content that comes before the first child header)
        section_content = '\n'.join(section.get('content', [])).strip()
        if section_content:
            section_content = self._get_header_prefix(section) + '\n\n' + section_content
            chunks.append(Document(
                page_content=section_content,
                metadata=section['metadata']
            ))
            
        # Process each child section individually
        for child in section['children']:
            # Try to create a chunk with the child and all its descendants
            child_content = self._section_to_content(child)
            token_count = self.count_tokens(child_content)
            
            if token_count <= self.token_limit:
                # Child fits in one chunk
                chunks.append(Document(
                    page_content=child_content.strip(),
                    metadata=child['metadata']
                ))
            else:
                # Child is too large, recursively process it
                sub_chunks = self._recursive_chunk_section(child, current_level + 1)
                chunks.extend(sub_chunks)

        return chunks


    def _parse_header_structure(self, doc: Document) -> Dict:
        """Parse document into hierarchical header structure."""
        lines = doc.page_content.split('\n')
        header_pattern = re.compile(r'^(#{1,6})\s+(.*)')
        code_block_pattern = re.compile(r'^\s*```')
        in_code_block = False
        
        root = {
            'level': 0,
            'title': 'root',
            'content': [],
            'children': [],
            'metadata': doc.metadata.copy(),
            'header_path': {}
        }
        
        stack = [root]  # Stack to track current hierarchy
        current_content = []
        
        for line in lines:
            if code_block_pattern.match(line):
                in_code_block = not in_code_block
            
            if not in_code_block:
                match = header_pattern.match(line)
                if match:
                    # Save accumulated content to current section
                    if current_content:
                        stack[-1]['content'].extend(current_content)
                        current_content = []
                    
                    # Extract header info
                    header_level = len(match.group(1))
                    header_text = match.group(2).strip()
                    
                    # Clean header text
                    header_text = re.sub(r'\\', '', header_text)
                    header_text = re.sub(r'\[¶\]\(.*?\)', '', header_text).strip()
                    
                    # Pop stack until we find the right parent level
                    while len(stack) > 1 and stack[-1]['level'] >= header_level:
                        parent = stack.pop()
                    
                    # Build header path
                    header_path = stack[-1]['header_path'].copy()
                    header_path[f'Header {header_level}'] = header_text
                    
                    # Create new section
                    section = {
                        'level': header_level,
                        'title': header_text,
                        'content': [],  # Include the header line itself
                        'children': [],
                        'metadata': {**doc.metadata, **header_path},
                        'header_path': header_path
                    }
                    
                    # Add to parent's children
                    stack[-1]['children'].append(section)
                    stack.append(section)
                else:
                    current_content.append(line)
            else:
                current_content.append(line)
        
        # Save accumulated content to last section
        if current_content:
            stack[-1]['content'].extend(current_content)
        
        with open('header_structure.json', 'w') as f:
            json.dump(root, f, indent=4)
        
        return root
    
    def _split_at_header_level(self, root_section: Dict, target_level: int) -> List[Document]:
        """Split document at specified header level."""
        chunks = []
        
        def collect_sections_at_level(section, current_level):
            if current_level == target_level:
                # Create chunk from this section and all its children
                content = self._section_to_content(section)
                if content.strip():
                    logger.info(f"[HeaderChunk] Level {target_level} chunk created with token count: {self.count_tokens(content)}")
                    chunks.append(Document(
                        page_content=content.strip(),
                        metadata=section['metadata']
                    ))
            else:
                # If we haven't reached target level, recurse into children
                if section['children']:
                    for child in section['children']:
                        collect_sections_at_level(child, child['level'])
                else:
                    # No children and not at target level - create chunk anyway
                    content = self._section_to_content(section)
                    if content.strip():
                        logger.info(f"[HeaderChunk] (No children, not at target) Level {current_level} chunk created with token count: {self.count_tokens(content)}")
                        chunks.append(Document(
                            page_content=content.strip(),
                            metadata=section['metadata']
                        ))
        # Handle root level specially
        if target_level == 1:
            # Split at level 1 headers
            for child in root_section['children']:
                if child['level'] <= target_level:
                    content = self._section_to_content(child)
                    if content.strip():
                        logger.info(f"[HeaderChunk] Level 1 chunk created with token count: {self.count_tokens(content)}")
                        chunks.append(Document(
                            page_content=content.strip(),
                            metadata=child['metadata']
                        ))
            # Add any root content that comes before first header
            if root_section['content']:
                root_content = '\n'.join(root_section['content']).strip()
                if root_content:
                    logger.info(f"[HeaderChunk] Root content chunk created with token count: {self.count_tokens(root_content)}")
                    chunks.insert(0, Document(
                        page_content=root_content,
                        metadata=root_section['metadata']
                    ))
        else:
            # For levels > 1, collect sections at that level
            for child in root_section['children']:
                collect_sections_at_level(child, child['level'])
        return chunks
    
    def _section_to_content(self, section: Dict, include_parent_content: bool = True) -> str:
        """Convert a section and all its children to content string, including ancestor headers."""
        content_lines = []

        # Add parent headers
        header_prefix = []
        for level in range(1, section["level"] + 1):
            header_text = section['header_path'].get(f'Header {level}')
            if header_text:
                header_prefix.append(f'{"#" * level} {header_text}')
        
        if header_prefix:
            content_lines.extend(header_prefix)

        if include_parent_content:
            content_lines.extend(section['content'])
       
        for child in section['children']:
            child_content = self._section_to_content(child)
            content_lines.extend(child_content.split('\n'))

        return '\n'.join(content_lines)

    def _process_oversized_chunk(self, chunk: Document) -> List[Document]:
        """Process chunks that are still too large after header splitting."""
        # Check if chunk contains tables
        if self._contains_table(chunk.page_content):
            return self._split_table_content(chunk)
        else:
            # Step 3: Split by paragraphs
            return self._split_by_paragraphs(chunk)
    
    def _contains_table(self, content: str) -> bool:
        """Check if content contains markdown tables."""
        table_pattern = re.compile(r'^\|.*\|$', re.MULTILINE)
        return bool(table_pattern.search(content))
    
    def _split_table_content(self, chunk: Document) -> List[Document]:
        """Handle table splitting."""
        lines = chunk.page_content.split('\n')
        chunks = []
        current_content = ""
        i = 0

        while i < len(lines):
            line = lines[i]
            if re.match(r'^\|.*\|$', line):
                table_end = self._find_table_end(lines, i)
                table_lines = lines[i:table_end + 1]
                table_str = '\n'.join(table_lines)

                # Step 1: Handle pre-table content
                pre_table_chunks = []
                if current_content.strip():
                    if self.count_tokens(current_content) <= self.token_limit:
                        pre_table_chunks.append(Document(
                            page_content=current_content.strip(),
                            metadata=chunk.metadata
                        ))
                    else:
                        para_chunks = self._split_by_paragraphs(
                            Document(current_content.strip(), chunk.metadata)
                        )
                        pre_table_chunks.extend(para_chunks)
                    current_content = ""

                # Step 2: Handle table content
                table_chunks = self._process_table(lines, i, chunk.metadata)

                # Step 3: Try merging pre_table[-1] + table[0]
                if pre_table_chunks and table_chunks:
                    combined = pre_table_chunks[-1].page_content + '\n' + table_chunks[0].page_content
                    if self.count_tokens(combined) <= self.token_limit:
                        logger.info("[TableChunk] Merging last pre-table with first table chunk")
                        pre_table_chunks[-1] = Document(
                            page_content=combined,
                            metadata=chunk.metadata
                        )
                        table_chunks = table_chunks[1:]

                # Append processed
                chunks.extend(pre_table_chunks)
                chunks.extend(table_chunks)

                i = table_end + 1
            else:
                current_content += line + '\n'
                i += 1

        # Step 4: Handle remaining post-table content
        if current_content.strip():
            if self.count_tokens(current_content) <= self.token_limit:
                post_table_chunks = [Document(
                    page_content=current_content.strip(),
                    metadata=chunk.metadata
                )]
            else:
                post_table_chunks = self._split_by_paragraphs(
                    Document(current_content.strip(), chunk.metadata)
                )
        else:
            post_table_chunks = []

        # Step 5: Try merging last table + first post-table
        if chunks and post_table_chunks:
            if re.match(r'^\|.*\|$', chunks[-1].page_content.strip().split('\n')[0]):
                combined = chunks[-1].page_content + '\n' + post_table_chunks[0].page_content
                if self.count_tokens(combined) <= self.token_limit:
                    logger.info("[TableChunk] Merging last table with first post-table chunk")
                    chunks[-1] = Document(
                        page_content=combined,
                        metadata=chunk.metadata
                    )
                    post_table_chunks = post_table_chunks[1:]

        chunks.extend(post_table_chunks)
        return chunks

    
    def _process_table(self, lines: List[str], start_idx: int, metadata: Dict) -> List[Document]:
        """Process a single table according to the table splitting rules."""
        
        # Extract table content
        table_end = self._find_table_end(lines, start_idx)
        table_lines = lines[start_idx:table_end + 1]
        
        # Parse table
        header_row = table_lines[0] if table_lines else ""
        separator_row = table_lines[1] if len(table_lines) > 1 else ""
        data_rows = table_lines[2:] if len(table_lines) > 2 else []
        
        # Rule 1: Try whole table 
        full_table = "\n".join(table_lines)
        if self.count_tokens(full_table) <= self.token_limit:
            return [Document(page_content=full_table.strip(), metadata=metadata)]
        
        # Rule 2: Split by rows, keep header 
        chunks = []
        base_content = f"{header_row}\n{separator_row}"
        
        current_chunk = base_content
        for row in data_rows:
            test_content = current_chunk + "\n" + row
            if self.count_tokens(test_content) <= self.token_limit:
                current_chunk = test_content
            else:
                # Save current chunk and start new one
                if current_chunk != base_content:
                    chunks.append(Document(page_content=current_chunk.strip(), metadata=metadata))
                
                # Check if single row fits
                single_row_content = base_content + "\n" + row
                if self.count_tokens(single_row_content) <= self.token_limit:
                    current_chunk = single_row_content
                else:
                    # Rule 3: Split row by cells
                    cell_chunks = self._split_table_row(header_row, row, metadata)
                    chunks.extend(cell_chunks)
                    current_chunk = base_content
        
        # Add final chunk
        if current_chunk != base_content:
            chunks.append(Document(page_content=current_chunk.strip(), metadata=metadata))
        
        return chunks
    
    def _split_table_row(self, header_row: str, data_row: str, metadata: Dict) -> List[Document]:
        """Split a table row by cells when the entire row doesn't fit."""
        headers = [h.strip() for h in header_row.split('|')[1:-1]]  # Remove empty first/last
        cells = [c.strip() for c in data_row.split('|')[1:-1]]
        
        chunks = []
        current_headers = []
        current_cells = []
        
        for i, (header, cell) in enumerate(zip(headers, cells)):
            test_headers = current_headers + [header]
            test_cells = current_cells + [cell]
            
            # Create test content
            test_table = self._create_mini_table(test_headers, test_cells)
            
            if self.count_tokens(test_table) <= self.token_limit:
                current_headers = test_headers
                current_cells = test_cells
            else:
                # Save current chunk if not empty
                if current_headers:
                    chunk_table = self._create_mini_table(current_headers, current_cells)
                    chunks.append(Document(page_content=chunk_table.strip(), metadata=metadata))
                
                # Rule 4: If single cell doesn't fit, split by paragraphs but add header
                single_cell_table = self._create_mini_table([header], [cell])
                if self.count_tokens(single_cell_table) <= self.token_limit:
                    current_headers = [header]
                    current_cells = [cell]
                else:
                    # Split cell content by paragraphs, adding column header
                    cell_chunks = self._split_cell_content(header, cell, metadata)
                    chunks.extend(cell_chunks)
                    current_headers = []
                    current_cells = []
        
        # Add final chunk
        if current_headers:
            chunk_table = self._create_mini_table(current_headers, current_cells)
            chunks.append(Document(page_content=chunk_table.strip(), metadata=metadata))
        
        return chunks
    
    def _create_mini_table(self, headers: List[str], cells: List[str]) -> str:
        """Create a mini table from headers and cells."""
        if not headers or not cells:
            return ""
        
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_row = "| " + " | ".join(cells) + " |"
        
        table = f"{header_row}\n{separator_row}\n{data_row}"
        
        return table
    
    def _split_cell_content(self, header: str, cell: str, metadata: Dict) -> List[Document]:
        """Split cell content by paragraphs, always adding column header."""
        paragraphs = [p.strip() for p in cell.split('\n\n') if p.strip()]
        chunks = []
        
        for para in paragraphs:
            content_with_header = f"Column: {header}\n\n{para}"
            
            if self.count_tokens(content_with_header) <= self.token_limit:
                chunks.append(Document(page_content=content_with_header, metadata=metadata))
            else:
                # Split by sentences
                sentence_chunks = self._split_by_sentences(
                    Document(content_with_header, metadata)
                )
                chunks.extend(sentence_chunks)
        
        return chunks
    
    def _find_table_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end index of a table."""
        i = start_idx
        while i < len(lines) and re.match(r'^\|.*\|$', lines[i]):
            i += 1
        return i - 1
    
    def _split_by_paragraphs(self, chunk: Document) -> List[Document]:
        """Step 3: Split by paragraphs."""
        paragraphs = [p.strip() for p in chunk.page_content.split('\n\n') if p.strip()]
        chunks = []
        current_content = ""
        
        for para in paragraphs:
            test_content = current_content + "\n\n" + para if current_content else para
            
            if self.count_tokens(test_content) <= self.token_limit:
                current_content = test_content
            else:
                # Save current content if not empty
                if current_content:
                    logger.info(f"[ParagraphChunk] Chunk created with token count: {self.count_tokens(current_content)}")
                    chunks.append(Document(
                        page_content=current_content.strip(),
                        metadata=chunk.metadata
                    ))
                # Check if single paragraph fits
                if self.count_tokens(para) <= self.token_limit:
                    current_content = para
                else:
                    # Step 4: Split by sentences
                    sentence_chunks = self._split_by_sentences(
                        Document(para, chunk.metadata)
                    )
                    for sc in sentence_chunks:
                        logger.info(f"[SentenceChunk] (from paragraph) Chunk created with token count: {self.count_tokens(sc.page_content)}")
                    chunks.extend(sentence_chunks)
                    current_content = ""
        # Add final chunk
        if current_content:
            logger.info(f"[ParagraphChunk] Final chunk created with token count: {self.count_tokens(current_content)}")
            chunks.append(Document(
                page_content=current_content.strip(),
                metadata=chunk.metadata
            ))
        return chunks
    
    def _split_by_sentences(self, chunk: Document) -> List[Document]:
        """Step 4: Split by sentences."""
        # Simple sentence splitting (can be improved with more sophisticated NLP)
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_pattern.split(chunk.page_content)
        
        chunks = []
        current_content = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_content = current_content + " " + sentence if current_content else sentence
            
            if self.count_tokens(test_content) <= self.token_limit:
                current_content = test_content
            else:
                # Save current content if not empty
                if current_content:
                    logger.info(f"[SentenceChunk] Chunk created with token count: {self.count_tokens(current_content)}")
                    chunks.append(Document(
                        page_content=current_content.strip(),
                        metadata=chunk.metadata
                    ))
                # If single sentence is too long, truncate or handle as needed
                if self.count_tokens(sentence) <= self.token_limit:
                    current_content = sentence
                else:
                    # Handle extremely long sentences by word splitting
                    word_chunks = self._split_by_words(
                        Document(sentence, chunk.metadata)
                    )
                    for wc in word_chunks:
                        logger.info(f"[WordChunk] (from sentence) Chunk created with token count: {self.count_tokens(wc.page_content)}")
                    chunks.extend(word_chunks)
                    current_content = ""
        # Add final chunk
        if current_content:
            logger.info(f"[SentenceChunk] Final chunk created with token count: {self.count_tokens(current_content)}")
            chunks.append(Document(
                page_content=current_content.strip(),
                metadata=chunk.metadata
            ))
        return chunks
    
    def _split_by_words(self, chunk: Document) -> List[Document]:
        """Emergency fallback: split by words when sentences are too long."""
        words = chunk.page_content.split()
        chunks = []
        current_content = ""
        
        for word in words:
            test_content = current_content + " " + word if current_content else word
            
            if self.count_tokens(test_content) <= self.token_limit:
                current_content = test_content
            else:
                if current_content:
                    logger.info(f"[WordChunk] Chunk created with token count: {self.count_tokens(current_content)}")
                    chunks.append(Document(
                        page_content=current_content.strip(),
                        metadata=chunk.metadata
                    ))
                current_content = word
        if current_content:
            logger.info(f"[WordChunk] Final chunk created with token count: {self.count_tokens(current_content)}")
            chunks.append(Document(
                page_content=current_content.strip(),
                metadata=chunk.metadata
            ))
        return chunks
    
    def _chunks_to_json(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """Convert Document chunks to JSON format."""
        result = []
        for i, chunk in enumerate(chunks):
            
            logger.info(f"[FinalChunk] Chunk {i} with token count: {self.count_tokens(chunk.page_content)}")
            result.append({
                'chunk_id': i,
                'content': chunk.page_content,
                'metadata': chunk.metadata,
                'token_count': self.count_tokens(chunk.page_content)
            })
        return result
    
    def save_chunks_to_json(self, chunks: List[Dict[str, Any]], output_path: str):
        """Save chunks to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    chunker = ExcelMarkdownChunker(token_limit=1000)
    chunks = chunker.chunk_from_file('data/markdown/input.md')
    chunker.save_chunks_to_json(chunks, 'data/chunk/output.json')

