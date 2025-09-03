#!/usr/bin/env python3
"""
Consolidated Markdown Chunker - Merges all 4 steps into a single class
Inherits from BaseMarkdownChunker and implements intelligent markdown chunking
"""
import os
import re
import uuid
import json
import chardet
import tiktoken
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.text_splitter import MarkdownHeaderTextSplitter

from src.core.markdown_chunker.base_markdown_chunker import BaseMarkdownChunker
from src.core.llm_client import LLMClientFactory
from src.core.markdown_chunker.prompts.markdown_chunker_prompts import (
    SYSTEM_MESSAGE, 
    CHUNK_START_MARKER, 
    CHUNK_END_MARKER,
    TABLE_SPLITTING_PROMPT_TEMPLATE,
    RETRY_WARNING_TEMPLATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants for LLM processing
LLM_MAX_TOKENS = 4000
LLM_TEMPERATURE = 0.0


class DocxMarkdownChunker(BaseMarkdownChunker):
    """
    Advanced Markdown Chunker that implements a 4-step process:
    1. Basic header-based chunking with table detection
    2. Merge small chunks and optimize hierarchy
    3. Split oversized chunks intelligently
    4. Process oversized table chunks with LLM
    """
    
    def __init__(self, max_tokens: int = 650, model_name: str = "gpt-4.1"):
        super().__init__()
        self.max_tokens = max_tokens
        
        # Step 1: Header splitter
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"), ("##", "H2"), ("###", "H3"), 
                ("####", "H4"), ("#####", "H5"), ("######", "H6")
            ],
            strip_headers=False  # Keep headers in content to preserve text
        )
        
        # Step 4: LLM client for table processing
        self._initialize_llm_client(model_name)

    def _initialize_llm_client(self, model_name: str):
        """Initialize the LLM client for table processing."""
        try:
            self.llm_factory = LLMClientFactory()
            self.llm_client = self.llm_factory.get_client(model_name)
            logger.info(f"✅ {model_name} client initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize GPT client: {e}. Table splitting will be skipped.")
            self.llm_client = None

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if not text:
            return 0
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def _generate_table_splitting_prompt(self, max_tokens: int, content_tokens: int, content: str, retry: bool = False) -> str:
        """
        Generate prompt for LLM to clean and split markdown content with tables.
        
        Args:
            max_tokens: Maximum tokens allowed per chunk
            content_tokens: Number of tokens in the input content
            content: The content to be processed
            retry: Whether this is a retry attempt (adds warning)
        
        Returns:
            The formatted prompt string
        """
        retry_warning = ""
        if retry:
            retry_warning = RETRY_WARNING_TEMPLATE.format(max_tokens=max_tokens)
        
        estimated_words = int(max_tokens * 0.75)
        
        return TABLE_SPLITTING_PROMPT_TEMPLATE.format(
            max_tokens=max_tokens,
            estimated_words=estimated_words,
            chunk_start_marker=CHUNK_START_MARKER,
            chunk_end_marker=CHUNK_END_MARKER,
            retry_warning=retry_warning,
            content_tokens=content_tokens,
            content=content
        )

    # ============================================================================
    # STEP 1: Basic Header-based Chunking with Table Detection
    # ============================================================================
    
    def _is_markdown_table(self, text: str) -> bool:
        """Check if text contains a markdown table (both pipe and grid tables)."""
        lines = text.strip().splitlines()
        if len(lines) < 2:
            return False
        
        # Method 1: Check for grid tables (using +, -, =)
        grid_separators = [l for l in lines if 
                          l.strip() and 
                          all(c in "-+=| \t" for c in l.strip()) and
                          ('+' in l or '=' in l)]
        
        if len(grid_separators) >= 2:
            return True
        
        # Method 2: Check for pipe tables
        pipe_lines = [l for l in lines if "|" in l]
        if len(pipe_lines) < 2:
            return False
        
        # Look for pipe table separator line
        has_pipe_separator = any(
            all(c in "-| \t:" for c in line.strip()) and "|" in line and "-" in line
            for line in lines
        )
        
        # Method 3: Check for consistent pipe patterns
        if len(pipe_lines) >= 3:
            pipe_positions = []
            for line in pipe_lines[:5]:
                positions = [i for i, c in enumerate(line) if c == '|']
                pipe_positions.append(positions)
            
            if len(pipe_positions) >= 2:
                first_positions = set(pipe_positions[0])
                similar_count = sum(1 for pos_list in pipe_positions[1:] 
                                  if len(set(pos_list) & first_positions) >= len(first_positions) * 0.5)
                if similar_count >= 1:
                    return True
        
        return has_pipe_separator

    def _step1_basic_chunking(self, markdown_text: str) -> List[Dict[str, Any]]:
        """Step 1: Basic header-based chunking with table detection."""
        logger.info("🔄 Step 1: Basic header-based chunking")
        
        initial_chunks = self.header_splitter.split_text(markdown_text)
        final_chunks = []
        
        for i, chunk in enumerate(initial_chunks):
            content = chunk.page_content.strip()
            if not content:
                continue
            
            metadata = chunk.metadata.copy()
            metadata["original_order"] = i
            metadata["is_table"] = self._is_markdown_table(content)
            
            chunk_info = {
                "chunk_id": str(uuid.uuid4()),
                "content": content,
                "length": self.count_tokens(content),
                "metadata": metadata
            }
            
            final_chunks.append(chunk_info)
        
        logger.info(f"✅ Step 1: {len(final_chunks)} chunks")
        return final_chunks

    # ============================================================================
    # STEP 2: Merge and Optimize Chunks
    # ============================================================================
    
    def _is_empty_header_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Check if chunk only contains empty header with CSS attributes."""
        content = chunk['content'].strip()
        
        empty_header_patterns = [
            r'^#+\s*\{[^}]*\}\s*$',  # Headers with only CSS attributes
            r'^#+\s*$',               # Headers with no content at all
        ]
        
        for pattern in empty_header_patterns:
            if re.match(pattern, content):
                return True
        
        if len(content) < 30 and content.count('#') > 0 and len(content.replace('#', '').strip()) < 5:
            return True
            
        return False

    def _get_header_hierarchy(self, chunk: Dict[str, Any]) -> tuple:
        """Get header hierarchy as tuple for grouping."""
        metadata = chunk['metadata']
        return (
            metadata.get('H1', ''), metadata.get('H2', ''), metadata.get('H3', ''),
            metadata.get('H4', ''), metadata.get('H5', ''), metadata.get('H6', '')
        )

    def _get_chunk_level_and_parent(self, chunk: Dict[str, Any]) -> Tuple[int, tuple]:
        """
        Get the chunk's heading level and parent hierarchy.
        Returns: (level, parent_hierarchy)
        
        Example:
        - H1: a → (1, ())
        - H2: b under H1: a → (2, ('a',))  
        - H3: c under H1: a, H2: b → (3, ('a', 'b'))
        """
        hierarchy = self._get_header_hierarchy(chunk)
        
        # Find the deepest non-empty heading level
        level = 0
        for i, header in enumerate(hierarchy, 1):
            if header:
                level = i
        
        # If no headers, this is root level content
        if level == 0:
            return (0, ())
        
        # Check if this chunk contains parent headers with no content between them
        effective_level = self._get_effective_chunk_level(chunk, level, hierarchy)
        
        # Parent hierarchy is everything above effective level
        parent_hierarchy = hierarchy[:effective_level-1] if effective_level > 1 else ()
        
        return (effective_level, parent_hierarchy)

    def _get_effective_chunk_level(self, chunk: Dict[str, Any], original_level: int, hierarchy: tuple) -> int:
        """
        Determine the effective level of a chunk by analyzing its content.
        If a chunk contains parent headers with no content between them and child headers,
        the effective level should be the parent level.
        
        Returns the effective level of the chunk.
        """
        if original_level <= 1:
            return original_level
        
        content = chunk['content']
        lines = content.split('\n')
        
        # Find all header lines in the content
        header_lines = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith('#'):
                # Count the number of # characters
                header_level = 0
                for char in line_stripped:
                    if char == '#':
                        header_level += 1
                    else:
                        break
                
                # Extract header text
                header_text = line_stripped[header_level:].strip()
                header_lines.append({
                    'level': header_level,
                    'text': header_text,
                    'line_index': i
                })
        
        if len(header_lines) <= 1:
            return original_level
        
        # Check if headers are nested without content between them
        for i in range(len(header_lines) - 1):
            current_header = header_lines[i]
            next_header = header_lines[i + 1]
            
            # Check if there's meaningful content between headers
            start_line = current_header['line_index'] + 1
            end_line = next_header['line_index']
            
            has_meaningful_content = False
            for line_idx in range(start_line, end_line):
                if line_idx < len(lines):
                    line_content = lines[line_idx].strip()
                    # Ignore empty lines, horizontal rules, and simple formatting
                    if (line_content and 
                        not line_content.startswith('---') and 
                        not line_content.startswith('===') and
                        line_content != ''):
                        has_meaningful_content = True
                        break
            
            # If no meaningful content between parent and child header,
            # the effective level should be the parent level
            if (not has_meaningful_content and 
                current_header['level'] < next_header['level'] and
                current_header['level'] == original_level - 1):
                
                return current_header['level']
        
        return original_level

    def _can_merge_chunks_hierarchical(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> bool:
        """Check if two chunks can be merged: same level & parent OR parent-child relationship."""
        l1, p1 = self._get_chunk_level_and_parent(chunk1)
        l2, p2 = self._get_chunk_level_and_parent(chunk2)
        
        # Case 1: Same level and same parent
        if l1 == l2 and p1 == p2:
            return True
        
        # Case 2: Parent-child relationship (adjacent levels)
        if abs(l1 - l2) == 1:
            h1 = self._get_header_hierarchy(chunk1)
            h2 = self._get_header_hierarchy(chunk2)
            
            if l1 < l2:  # chunk1 is parent of chunk2
                # chunk2's parent should match chunk1's hierarchy up to chunk1's level
                return p2 == h1[:l1]
            else:  # chunk2 is parent of chunk1
                # chunk1's parent should match chunk2's hierarchy up to chunk2's level
                return p1 == h2[:l2]
        
        return False

    def _merge_chunk_contents(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple chunks into one."""
        if not chunks:
            return None
        
        if len(chunks) == 1:
            return chunks[0]
        
        combined_content = []
        combined_metadata = chunks[0]['metadata'].copy()
        is_any_table = False
        
        for chunk in chunks:
            combined_content.append(chunk['content'])
            if chunk['metadata'].get('is_table', False):
                is_any_table = True
        
        combined_metadata['is_table'] = is_any_table
        combined_metadata['merged_from'] = [c['metadata'].get('original_order', -1) for c in chunks]
        combined_metadata['merge_count'] = len(chunks)
        
        merged_chunk = {
            "chunk_id": str(uuid.uuid4()),
            "content": "\n\n".join(combined_content),
            "length": 0,  # Will be recalculated
            "metadata": combined_metadata
        }
        
        merged_chunk["length"] = self.count_tokens(merged_chunk["content"])
        return merged_chunk

    def _build_header_tree(self, chunks: List[Dict[str, Any]]):
        """Build tree of chunks based on heading hierarchy with strict parent validation."""
        root = {"level": 0, "chunk": None, "children": [], "parent": None}
        stack = [root]
        
        for i, chunk in enumerate(chunks):
            level, parent_hierarchy = self._get_chunk_level_and_parent(chunk)
            
            node = {"level": level, "chunk": chunk, "children": [], "parent": None}
            
            # Find the correct parent by validating header hierarchy
            found_parent = False
            while len(stack) > 1:  # Keep at least root
                potential_parent = stack[-1]
                
                if potential_parent["level"] >= level:
                    # Pop nodes that are at same or higher level
                    stack.pop()
                    continue
                
                # Check if this is a valid parent by comparing hierarchies
                if potential_parent.get("chunk"):
                    parent_chunk = potential_parent["chunk"]
                    parent_level, _ = self._get_chunk_level_and_parent(parent_chunk)
                    parent_full_hierarchy = self._get_header_hierarchy(parent_chunk)
                    
                    # For a valid parent-child relationship:
                    # 1. Parent level must be LESS than child level (không cần chính xác -1)
                    # 2. Child's parent hierarchy must START WITH parent's full hierarchy
                    if (parent_level < level and 
                        len(parent_hierarchy) >= parent_level and
                        parent_hierarchy[:parent_level] == parent_full_hierarchy[:parent_level]):
                        
                        found_parent = True
                        break
                    else:
                        stack.pop()
                        continue
                else:
                    # This is the root or a node without chunk
                    found_parent = True
                    break
            
            if not found_parent:
                # If no valid parent found, attach to root
                while len(stack) > 1:
                    stack.pop()
            
            parent = stack[-1]
            parent["children"].append(node)
            node["parent"] = parent
            stack.append(node)
        
        return root

    def _merge_hierarchical_bottom_up(self, node):
        """
        Merge từ level thấp lên cao với thứ tự ưu tiên đúng:
        1. Gộp siblings cùng level, cùng parent, không có children, cạnh nhau
        2. Chỉ gộp children với parent khi TẤT CẢ children đã được gộp thành 1 chunk
        """
        # Đệ quy xử lý tất cả children trước (bottom-up)
        for i, child in enumerate(node["children"]):
            self._merge_hierarchical_bottom_up(child)
        
        # Bước 1: Gộp các siblings cùng level không có children
        merged_children = []
        current_group = []
        sibling_merges = 0
        
        for i, child in enumerate(node["children"]):
            # Nếu child này không có children (là leaf) và có thể gộp với group hiện tại
            if not child["children"]:
                if not current_group:
                    current_group.append(child)
                else:
                    # Kiểm tra có thể gộp với group không
                    last_child = current_group[-1]
                    if self._can_merge_sibling_nodes(last_child, child):
                        total_tokens = sum(c["chunk"]["length"] for c in current_group) + child["chunk"]["length"]
                        if total_tokens <= self.max_tokens:
                            current_group.append(child)
                        else:
                            # Group đầy, finalize group hiện tại
                            if len(current_group) > 1:
                                sibling_merges += 1
                            merged_children.append(self._merge_node_group(current_group))
                            current_group = [child]
                    else:
                        # Không thể gộp, finalize group hiện tại
                        if len(current_group) > 1:
                            sibling_merges += 1
                        merged_children.append(self._merge_node_group(current_group))
                        current_group = [child]
            else:
                # Child có children, finalize group hiện tại và thêm child này
                if current_group:
                    if len(current_group) > 1:
                        sibling_merges += 1
                    merged_children.append(self._merge_node_group(current_group))
                    current_group = []
                merged_children.append(child)
        
        # Finalize group cuối cùng
        if current_group:
            if len(current_group) > 1:
                sibling_merges += 1
            merged_children.append(self._merge_node_group(current_group))
        
        node["children"] = merged_children
        
        # Bước 2: Nếu tất cả children đã được gộp thành 1 chunk duy nhất,
        # và node hiện tại có chunk, thì gộp children với parent
        if (len(node["children"]) == 1 and 
            node.get("chunk") is not None and 
            node["children"][0].get("chunk") is not None):
            
            parent_chunk = node["chunk"]
            child_chunk = node["children"][0]["chunk"]
            
            # Kiểm tra có thể gộp parent với child không
            total_tokens = parent_chunk["length"] + child_chunk["length"]
            if total_tokens <= self.max_tokens:
                # Gộp parent content với child content
                merged_chunk = self._merge_chunk_contents([parent_chunk, child_chunk])
                node["chunk"] = merged_chunk
                node["children"] = []  # Đánh dấu không còn children

    def _can_merge_sibling_nodes(self, node1, node2) -> bool:
        """Kiểm tra 2 sibling nodes có thể gộp không."""
        if not node1.get("chunk") or not node2.get("chunk"):
            return False
        
        chunk1 = node1["chunk"]
        chunk2 = node2["chunk"]
        
        # Kiểm tra cùng level và cùng parent hierarchy
        can_merge = self._can_merge_chunks_hierarchical(chunk1, chunk2)
        
        return can_merge

    def _merge_node_group(self, group):
        """Merge list of sibling nodes into a single node if possible."""
        if len(group) == 1:
            return group[0]
        merged_chunk = self._merge_chunk_contents([g["chunk"] for g in group])
        return {"level": group[0]["level"], "chunk": merged_chunk, "children": [], "parent": group[0]["parent"]}

    def _flatten_tree(self, node):
        """Return list of chunks in original document order."""
        result = []
        
        # If this node has a chunk, add it
        if node.get("chunk"):
            result.append(node["chunk"])
        
        # Process all children recursively
        for child in node["children"]:
            result.extend(self._flatten_tree(child))
        
        return result

    def merge_chunks_hierarchical(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Public method: merge chunks using proper hierarchical priority."""
        
        # Step 1: Clean empty headers
        cleaned_chunks = []
        skip_next = False
        empty_header_merges = 0
        
        for i, chunk in enumerate(chunks):
            if skip_next:
                skip_next = False
                continue
            if self._is_empty_header_chunk(chunk):
                if i + 1 < len(chunks):
                    merged = self._merge_chunk_contents([chunk, chunks[i+1]])
                    cleaned_chunks.append(merged)
                    skip_next = True
                    empty_header_merges += 1
            else:
                cleaned_chunks.append(chunk)

        # Step 2: Build hierarchy tree and merge from bottom up
        tree_root = self._build_header_tree(cleaned_chunks)
        
        self._merge_hierarchical_bottom_up(tree_root)
        
        merged_chunks = self._flatten_tree(tree_root)

        logger.info(f"✅ Hierarchical merge complete: {len(cleaned_chunks)} → {len(merged_chunks)} chunks")
        return merged_chunks

    def _can_merge_chunks_by_tokens(self, chunks: List[Dict[str, Any]]) -> bool:
        """Check if chunks can be merged based on token count."""
        if not chunks:
            return False
        
        total_tokens = sum(chunk['length'] for chunk in chunks)
        return total_tokens <= self.max_tokens

    def _step2_merge_chunks(self, step1_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 2: Merge chunks using bottom-up tree-based approach."""
        logger.info("🔄 Step 2: Merging chunks using bottom-up tree algorithm")
        
        # Use the new hierarchical merge method
        merged_chunks = self.merge_chunks_hierarchical(step1_chunks)
        
        logger.info(f"✅ Step 2: {len(step1_chunks)} → {len(merged_chunks)} chunks (tree-based merging)")
        return merged_chunks

    # ============================================================================
    # STEP 3: Split Oversized Chunks Intelligently
    # ============================================================================
    
    def _find_split_points(self, content: str) -> List[Tuple[int, str, int]]:
        """Find good split points in content - prioritize semantic boundaries."""
        split_points = []
        lines = content.split('\n')
        current_pos = 0
        
        in_toc = False  # Track Table of Contents
        toc_entry_continues = False
        
        for i, line in enumerate(lines):
            current_pos += len(line) + 1
            
            line_lower = line.lower().strip()
            line_stripped = line.strip()
            
            # TOC detection
            if 'table of contents' in line_lower or 'contents' in line_lower:
                in_toc = True
                toc_entry_continues = False
            elif line_stripped.startswith('#') and not line_stripped.startswith('['):
                in_toc = False
                toc_entry_continues = False
            
            if in_toc:
                is_complete_toc_entry = re.match(r'^\s*\[[^\]]+\]\s*\[[^\]]+\]\([^)]+\)', line_stripped)
                is_toc_start = re.match(r'^\s*\[[^\[\]]*$', line_stripped) and not line_stripped.endswith(']')
                is_toc_middle = toc_entry_continues and not line_stripped.startswith('[') and not line_stripped.endswith(')')
                is_toc_end = toc_entry_continues and (line_stripped.endswith(')') or '](' in line_stripped)
                
                if is_toc_start:
                    toc_entry_continues = True
                elif is_toc_end or is_complete_toc_entry:
                    toc_entry_continues = False
                
                if line_stripped == '' and i + 1 < len(lines):
                    next_non_empty = None
                    for j in range(i + 1, min(i + 3, len(lines))):
                        if lines[j].strip():
                            next_non_empty = lines[j].strip()
                            break
                    
                    if next_non_empty and not re.match(r'^\s*\[', next_non_empty):
                        in_toc = False
                        toc_entry_continues = False
            
            # Apply splitting rules
            if i < len(lines) - 1 and line_stripped == '' and not in_toc:
                split_points.append((current_pos, 'paragraph_break', 100))
            elif line_stripped.startswith('#') and not in_toc:
                split_points.append((current_pos - len(line) - 1, 'header', 90))
            elif in_toc and line_stripped == '' and not toc_entry_continues:
                next_non_empty = None
                for j in range(i + 1, min(i + 4, len(lines))):
                    if lines[j].strip():
                        next_non_empty = lines[j].strip()
                        break
                
                if next_non_empty and not re.match(r'^\s*\[', next_non_empty):
                    split_points.append((current_pos, 'toc_end', 95))
            elif (re.match(r'^\s*[-*+]\s', line_stripped) or re.match(r'^\s*\d+\.\s', line_stripped)) and not in_toc:
                split_points.append((current_pos - len(line) - 1, 'list_item', 80))
            elif re.search(r'[.!?]\s*$', line_stripped) and not in_toc and not toc_entry_continues:
                split_points.append((current_pos, 'sentence_end', 70))
            elif line_stripped.endswith(':') and not in_toc and not toc_entry_continues:
                split_points.append((current_pos, 'colon_end', 60))
            elif not in_toc and not toc_entry_continues:
                split_points.append((current_pos, 'line_break', 30))
            elif in_toc and not toc_entry_continues:
                split_points.append((current_pos, 'toc_safe_break', 15))
        
        split_points.sort(key=lambda x: x[0])
        return split_points

    def _find_best_split(self, content: str, target_tokens: int) -> Optional[int]:
        """Find the best split point near target token count."""
        if self.count_tokens(content) <= self.max_tokens:
            return None
        
        split_points = self._find_split_points(content)
        if not split_points:
            return None
        
        target_chars = len(content) * target_tokens // self.count_tokens(content)
        
        best_split = None
        best_score = -1
        
        for pos, split_type, priority in split_points:
            if pos < target_chars * 0.3 or pos > target_chars * 1.5:
                continue
            
            distance_factor = 1 - abs(pos - target_chars) / target_chars
            score = priority * distance_factor
            
            if score > best_score:
                best_score = score
                best_split = pos
        
        if best_split is not None:
            best_split = self._adjust_to_line_boundary(content, best_split)
        
        return best_split

    def _adjust_to_line_boundary(self, content: str, position: int) -> int:
        """Adjust split position to ensure it's at a line boundary."""
        if position <= 0 or position >= len(content):
            return position
        
        prev_newline = content.rfind('\n', 0, position)
        next_newline = content.find('\n', position)
        
        if prev_newline == -1:
            return 0 if next_newline == -1 else next_newline + 1
        
        if next_newline == -1:
            return len(content)
        
        if position - prev_newline <= next_newline - position:
            return prev_newline + 1
        else:
            return next_newline + 1

    def _split_content_intelligently(self, content: str) -> List[str]:
        """Split content into smaller pieces while preserving semantic integrity."""
        if self.count_tokens(content) <= self.max_tokens:
            return [content]
        
        parts = []
        remaining = content
        
        while self.count_tokens(remaining) > self.max_tokens:
            target_tokens = int(self.max_tokens * 0.75)
            
            split_pos = self._find_best_split(remaining, target_tokens)
            
            if split_pos is None or split_pos <= 0:
                target_chars = len(remaining) * target_tokens // self.count_tokens(remaining)
                split_pos = self._adjust_to_line_boundary(remaining, target_chars)
            
            first_part = remaining[:split_pos].strip()
            
            if first_part:
                first_part_tokens = self.count_tokens(first_part)
                
                if first_part_tokens > self.max_tokens:
                    target_tokens = int(self.max_tokens * 0.6)
                    split_pos = self._find_best_split(remaining, target_tokens)
                    
                    if split_pos is None or split_pos <= 0:
                        target_chars = len(remaining) * target_tokens // self.count_tokens(remaining)
                        split_pos = self._adjust_to_line_boundary(remaining, target_chars)
                        split_pos = max(1, split_pos)
                    
                    first_part = remaining[:split_pos].strip()
                    first_part_tokens = self.count_tokens(first_part)
                
                if first_part_tokens <= self.max_tokens:
                    parts.append(first_part)
                else:
                    safe_chars = len(remaining) * self.max_tokens // (2 * self.count_tokens(remaining))
                    lines = remaining[:safe_chars].split('\n')
                    if len(lines) > 1:
                        safe_part = '\n'.join(lines[:-1]).strip()
                    else:
                        safe_part = remaining[:safe_chars].strip()
                    
                    if safe_part and self.count_tokens(safe_part) <= self.max_tokens:
                        parts.append(safe_part)
                        split_pos = len(safe_part)
                    else:
                        split_pos = max(1, len(remaining) // 3)
                        parts.append(remaining[:split_pos].strip())
            
            remaining = remaining[split_pos:].strip()
            
            if not remaining or len(parts) > 20:
                break
        
        if remaining.strip():
            parts.append(remaining.strip())
        
        return parts

    def _create_split_chunk(self, original_chunk: Dict[str, Any], content: str, part_index: int, total_parts: int) -> Dict[str, Any]:
        """Create a new chunk from split content."""
        new_metadata = original_chunk['metadata'].copy()
        new_metadata['split_from'] = original_chunk['chunk_id']
        new_metadata['part'] = f"{part_index + 1}/{total_parts}"
        new_metadata['original_length'] = original_chunk['length']
        
        # Preserve header information
        for key in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
            if key in original_chunk['metadata']:
                new_metadata[key] = original_chunk['metadata'][key]
        
        return {
            "chunk_id": str(uuid.uuid4()),
            "content": content,
            "length": self.count_tokens(content),
            "metadata": new_metadata
        }

    def _step3_split_oversized(self, step2_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 3: Split chunks that exceed max_tokens."""
        logger.info("🔄 Step 3: Splitting oversized chunks")
        
        split_chunks = []
        oversized_count = 0
        split_count = 0
        preserved_table_count = 0
        
        for i, chunk in enumerate(step2_chunks):
            if chunk['length'] > self.max_tokens:
                oversized_count += 1
                
                # Skip splitting if chunk contains a table - will be handled in step 4
                if chunk['metadata'].get('is_table', False):
                    split_chunks.append(chunk)
                    preserved_table_count += 1
                    continue
                
                content_parts = self._split_content_intelligently(chunk['content'])
                
                if len(content_parts) > 1:
                    split_count += 1
                    
                    for j, part_content in enumerate(content_parts):
                        split_chunk = self._create_split_chunk(chunk, part_content, j, len(content_parts))
                        split_chunks.append(split_chunk)
                else:
                    split_chunks.append(chunk)
            else:
                split_chunks.append(chunk)
        
        logger.info(f"✅ Step 3: {len(split_chunks)} chunks")
        return split_chunks

    # ============================================================================
    # STEP 4: LLM-based Table Splitting
    # ============================================================================
    
    def _generate_llm_prompt(self, content: str, content_tokens: int, retry: bool = False) -> str:
        """Generate prompt for LLM to clean and split content."""
        return self._generate_table_splitting_prompt(self.max_tokens, content_tokens, content, retry)

    def _parse_chunks_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse chunks from LLM response using markers."""
        pattern = f"{CHUNK_START_MARKER}\\s*(.*?)\\s*{CHUNK_END_MARKER}"
        chunks_text = re.findall(pattern, response_text, re.DOTALL)
        result = []
        for chunk_text in chunks_text:
            token_count = self.count_tokens(chunk_text)
            result.append({"content": chunk_text.strip(), "length": token_count})
        return result

    def _split_with_llm(self, content: str, retry: bool = False) -> List[Dict[str, Any]]:
        """Split content using LLM."""
        if not self.llm_client:
            raise RuntimeError("LLM client not available")
            
        content_tokens = self.count_tokens(content)
        prompt = self._generate_llm_prompt(content, content_tokens, retry)
        
        try:
            response = self.llm_client["client"].chat.completions.create(
                model=self.llm_client["model"],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ]
            )

            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError("LLM returned empty response")

            return self._parse_chunks_from_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"❌ Error during LLM processing: {e}")
            raise

    def _create_chunks_from_llm_content(self, original_chunk: Dict[str, Any], content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create new chunk objects from LLM split content."""
        new_chunks = []
        total_parts = len(content_list)
        
        for i, content_item in enumerate(content_list):
            metadata = original_chunk['metadata'].copy()
            metadata.update({
                "split_from": original_chunk['chunk_id'],
                "part": f"{i+1}/{total_parts}",
                "original_length": original_chunk['length'],
                "split_method": "llm_table_split"
            })
            
            new_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "content": content_item["content"],
                "length": content_item["length"],
                "metadata": metadata
            })
            
        return new_chunks

    def _step4_split_table_chunks(self, step3_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 4: Split oversized table chunks using LLM."""
        if not self.llm_client:
            logger.warning("⚠️ Step 4: LLM client not available, skipping table splitting")
            return step3_chunks
            
        logger.info("🔄 Step 4: Processing oversized table chunks with LLM")
        
        split_chunks = []
        
        for i, chunk in enumerate(step3_chunks):
            if chunk["length"] > self.max_tokens and chunk['metadata'].get('is_table', False):
                try:
                    split_content = self._split_with_llm(chunk["content"])
                    new_chunks = self._create_chunks_from_llm_content(chunk, split_content)
                    split_chunks.extend(new_chunks)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to split table chunk {i+1}: {e}")
                    split_chunks.append(chunk)  # Keep original if splitting fails
            else:
                split_chunks.append(chunk)
        
        logger.info(f"✅ Step 4: {len(split_chunks)} chunks")
        return split_chunks

    # ============================================================================
    # MAIN PROCESSING PIPELINE
    # ============================================================================
    
    def _read_file(self, file_path: str) -> str:
        """Read file with encoding detection."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            encoding_info = chardet.detect(raw_data)
            encoding = encoding_info.get('encoding', 'utf-8')
        
        # Read with detected encoding
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to utf-8 with error handling
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        return content

    def process_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process file through all 4 steps and return intermediate results.
        Returns: (step1_chunks, step2_chunks, step3_chunks, step4_chunks)
        """
        logger.info(f"🚀 Processing: {os.path.basename(file_path)}")
        
        # Read file content
        content = self._read_file(file_path)
        
        # Step 1: Basic header-based chunking
        step1_chunks = self._step1_basic_chunking(content)
        
        # Step 2: Merge and optimize chunks
        step2_chunks = self._step2_merge_chunks(step1_chunks)
        
        # Step 3: Split oversized chunks
        step3_chunks = self._step3_split_oversized(step2_chunks)
        
        # Step 4: LLM-based table splitting
        step4_chunks = self._step4_split_table_chunks(step3_chunks)
        
        logger.info(f"✅ Completed: {len(step4_chunks)} final chunks")
        return step1_chunks, step2_chunks, step3_chunks, step4_chunks

    def process_content(self, content: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process content string directly through all 4 steps.
        Returns: (step1_chunks, step2_chunks, step3_chunks, step4_chunks)
        """
        logger.info(f"🚀 Processing content ({len(content)} chars)")
        
        # Step 1: Basic header-based chunking
        step1_chunks = self._step1_basic_chunking(content)
        
        # Step 2: Merge and optimize chunks
        step2_chunks = self._step2_merge_chunks(step1_chunks)
        
        # Step 3: Split oversized chunks
        step3_chunks = self._step3_split_oversized(step2_chunks)
        
        # Step 4: LLM-based table splitting
        step4_chunks = self._step4_split_table_chunks(step3_chunks)
        
        logger.info(f"✅ Completed: {len(step4_chunks)} final chunks")
        return step1_chunks, step2_chunks, step3_chunks, step4_chunks

    def format_chunks_for_output(self, chunks: List[Dict[str, Any]], file_name: str) -> List[Dict[str, Any]]:
        """
        Format chunks for output with the requested structure:
        {
            "chunk_id": "d1b03460bd674d39904d2d1f41d085a2",
            "embedding": "None",
            "lenght": "500",
            "metadata": {
                "H1": "RFP for Conduit Road Schoolhouse",
                "H2": "Proposal Selection Criteria", 
                "H3": "Criterion 1: Use",
                "file_name": "elpasoamiandmdmsrfpv19_10012019-final.md",
                "position": 1
            },
            "chunk_text": "abc"
        }
        """
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Extract headers from metadata
            headers = {}
            if 'H1' in chunk['metadata']:
                headers['H1'] = chunk['metadata']['H1']
            if 'H2' in chunk['metadata']:
                headers['H2'] = chunk['metadata']['H2']
            if 'H3' in chunk['metadata']:
                headers['H3'] = chunk['metadata']['H3']
            if 'H4' in chunk['metadata']:
                headers['H4'] = chunk['metadata']['H4']
            if 'H5' in chunk['metadata']:
                headers['H5'] = chunk['metadata']['H5']
            if 'H6' in chunk['metadata']:
                headers['H6'] = chunk['metadata']['H6']
            
            # Add file_name and position to metadata
            headers['file_name'] = file_name
            headers['position'] = i + 1
            
            formatted_chunk = {
                "chunk_id": chunk['chunk_id'],
                "embedding": "None",
                "lenght": str(chunk['length']),  # Note: keeping "lenght" as requested (typo preserved)
                "metadata": headers,
                "chunk_text": chunk['content']
            }
            
            formatted_chunks.append(formatted_chunk)
        
        return formatted_chunks


    def chunk_from_file(self, file_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Chunk a DOCX file into smaller parts, which have the meaning of paragraphs.
        Returns a list of text chunks, or save them to a json file
        
        Args:
            file_path: Path to the markdown file to chunk
            output_path: Optional path to save chunks as JSON file
            
        Returns:
            List of formatted chunks ready for use
        """
        logger.info(f"🚀 Processing: {os.path.basename(file_path)}")
        
        # Read file content
        content = self._read_file(file_path)
        
        # Step 1: Basic header-based chunking
        step1_chunks = self._step1_basic_chunking(content)
        
        # Step 2: Merge and optimize chunks
        step2_chunks = self._step2_merge_chunks(step1_chunks)
        
        # Step 3: Split oversized chunks
        step3_chunks = self._step3_split_oversized(step2_chunks)
        
        # Step 4: LLM-based table splitting
        step4_chunks = self._step4_split_table_chunks(step3_chunks)
        
        # Format chunks for output
        base_name = os.path.basename(file_path)
        formatted_chunks = self.format_chunks_for_output(step4_chunks, base_name)
        
        # Save to file if output_path is provided
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(formatted_chunks, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved: {output_path}")
        
        # Log summary statistics
        oversized_count = sum(1 for chunk in step4_chunks if chunk['length'] > self.max_tokens)
        token_counts = [chunk['length'] for chunk in step4_chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        max_tokens_actual = max(token_counts) if token_counts else 0
        min_tokens_actual = min(token_counts) if token_counts else 0
        
        logger.info(f"📊 {len(formatted_chunks)} chunks | Tokens: {min_tokens_actual}-{max_tokens_actual} (avg: {avg_tokens:.0f})")
        
        if oversized_count > 0:
            logger.warning(f"⚠️ {oversized_count} chunks exceed {self.max_tokens} tokens")
        else:
            logger.info(f"✅ All chunks within {self.max_tokens} token limit")
        
        logger.info(f"✅ Completed: {len(formatted_chunks)} final chunks")
        return formatted_chunks
    
if __name__ == "__main__":
    chunker = DocxMarkdownChunker()
    file_path = "data/sample_documents/elpasoamiandmdmsrfpv19_10012019-final_described_llm_converted.md"
    output_path = "data/output/chunks/elpasoamiandmdmsrfpv19_10012019-final.json"
    chunker.chunk_from_file(file_path, output_path)