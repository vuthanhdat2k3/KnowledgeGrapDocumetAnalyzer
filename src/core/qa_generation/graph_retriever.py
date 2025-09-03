#!/usr/bin/env python3

import logging
from typing import Dict, Any, List, Optional
import time
import re

from src.core.knowledge_graph.neo4j_manager import neo4j_manager
from src.core.llm_client import LLMClientFactory
from src.core.qa_generation.prompts import (
    VALIDATE_DOCUMENT_QUERY,
    LANGUAGE_DETECTION_SCOPED_QUERY,
    LANGUAGE_DETECTION_GLOBAL_QUERY,
    HYBRID_SEARCH_SCOPED_QUERY,
    HYBRID_SEARCH_GLOBAL_QUERY,
    PARENT_CONTENT_QUERY,
    RELATED_CONTENT_QUERY,
    SEMANTIC_SEARCH_SCOPED_QUERY,
    SEMANTIC_SEARCH_GLOBAL_QUERY,
    LANGUAGE_DETECTION_PROMPT,
    TRANSLATION_PROMPT,
    KEYWORD_EXTRACTION_PROMPT
)

# Setup logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

class GraphRetriever:
    
    def __init__(self, document_name: Optional[str] = None):
        """
        Initialize the retriever with optional document scope
        
        Args:
            document_name: Name of the document to scope searches to. 
                          If None, searches across entire graph.
        """
        self.logger = logger
        self.neo4j_manager = neo4j_manager
        self.llm_factory = None
        self.document_name = document_name
        
        # Setup connections
        self._neo4j_connection()
        self._setup_llm_client()
        
        # Validate document exists if document_name is provided
        if self.document_name and not self._validate_document_exists(self.document_name):
            raise ValueError(f"Document with file_name '{self.document_name}' not found in the knowledge graph")
        
        if self.document_name:
            self.logger.info(f"GraphRetriever initialized with document scope: {self.document_name}")
        else:
            self.logger.info("GraphRetriever initialized with global scope (entire graph)")
    
    def _neo4j_connection(self):
        """Setup Neo4j database connection"""
        try:
            if not self.neo4j_manager.is_connected():
                logger.info("Connecting to Neo4j database...")
                self.neo4j_manager.connect()
                logger.info("Successfully connected to Neo4j")
            else:
                logger.info("Neo4j connection already established")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def _setup_llm_client(self):
        """Setup LLM client for embeddings and answer generation"""
        try:
            self.llm_factory = LLMClientFactory()
            logger.info("LLM client setup successfully")
        except Exception as e:
            logger.error(f"Failed to setup LLM client: {e}")
            raise

    def _validate_document_exists(self, document_name: str) -> bool:
        """
        Validate that a document with the given file_name exists in the knowledge graph
        
        Args:
            document_name: The file_name to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            results = self.neo4j_manager.execute_query(VALIDATE_DOCUMENT_QUERY, {"document_name": document_name})
            return results[0]["document_count"] > 0 if results else False
        except Exception as e:
            logger.error(f"Error validating document existence: {e}")
            return False

    def detect_data_language(self, sample_size: int = 20) -> str:
        try:
            # Query to get sample content from various node types
            if self.document_name:
                # Scoped to specific document
                query = LANGUAGE_DETECTION_SCOPED_QUERY
                parameters = {
                    "sample_size": sample_size,
                    "document_name": self.document_name
                }
            else:
                # Global scope - entire graph
                query = LANGUAGE_DETECTION_GLOBAL_QUERY
                parameters = {"sample_size": sample_size}
            
            results = self.neo4j_manager.execute_query(query, parameters)
            
            if not results:
                return "Unknown"
            
            # Use all available samples (not just first 5)
            sample_texts = [result["content"] for result in results]
            combined_text = "\n\n".join(sample_texts)
            
            # Use LLM to detect language
            client_config = self.llm_factory.get_client("gpt-4.1")
            client = client_config.get('client') if isinstance(client_config, dict) else client_config
            
            prompt = LANGUAGE_DETECTION_PROMPT.format(combined_text=combined_text[:2000])
            
            response = client.chat.completions.create(
                model=self.llm_factory.models.get("gpt-4.1"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            detected_language = response.choices[0].message.content.strip()
            
            # Validate response - ensure it's a single word language name
            if detected_language and len(detected_language.split()) == 1:
                self.logger.info(f"Detected language: {detected_language}")
                return detected_language
            else:
                self.logger.warning(f"Invalid language response: {detected_language}")
                return "Unknown"
                
        except Exception as e:
            self.logger.error(f"Error detecting data language: {e}")
            return "Unknown"

    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a text using the configured embedding model
        
        Args:
            text: Text to create embedding for
            
        Returns:
            Embedding vector as list of floats, empty list if failed
        """
        try:
            client_config = self.llm_factory.get_client("embedding")
            client = client_config.get('client') if isinstance(client_config, dict) else client_config
            
            response = client.embeddings.create(
                input=text,
                model=self.llm_factory.models.get("embedding")
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            self.logger.error(f"Error creating embedding: {e}")
            return []

    def translate_question(self, question: str, target_language: str = "en") -> str:
        """
        Translate a question to target language for better semantic search
        
        Args:
            question: Original question text
            target_language: Target language code (default: "en")
            
        Returns:
            Translated question, or original question if translation fails
        """
        try:
            client_config = self.llm_factory.get_client("gpt-4.1")
            client = client_config.get('client') if isinstance(client_config, dict) else client_config
            
            prompt = TRANSLATION_PROMPT.format(question=question, target_language=target_language)
            
            response = client.chat.completions.create(
                model=self.llm_factory.models.get("gpt-4.1"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error translating question: {e}")
            return question  # Return original if translation fails

    def search_knowledge_graph_hybrid(self, 
                                      question_embedding: list,
                                      limit: int = 10,
                                      min_score: float = 0.1) -> List[Dict[str, Any]]:
        """
        Hybrid search combining multiple methods for better accuracy
        
        Args:
            question_embedding: Pre-computed embedding for the question (required)
            limit: Maximum number of results
            min_score: Minimum score threshold
            
        Returns:
            List of relevant nodes with metadata
        """
        try:
            # Validate that embedding is provided
            if not question_embedding:
                raise ValueError("question_embedding is required and cannot be empty")
            
            self.logger.info("Using pre-computed embedding for hybrid search")
            
            # Use embedding directly
            embedding = question_embedding
            
            # For keyword extraction, we'll use a simple approach since we don't have question text
            # We'll rely more heavily on semantic similarity
            keywords = []  # Empty keywords for pure semantic search
            
            # Hybrid query combining semantic similarity and keyword matching
            if self.document_name:
                # Document-scoped search
                query = HYBRID_SEARCH_SCOPED_QUERY
                search_method_suffix = "_document_scoped"
                parameters = {
                    "embedding": embedding,
                    "keywords": keywords,
                    "limit": limit,
                    "min_score": min_score,
                    "document_name": self.document_name
                }
            else:
                # Global search - entire graph
                query = HYBRID_SEARCH_GLOBAL_QUERY
                search_method_suffix = "_global_scoped"
                parameters = {
                    "embedding": embedding,
                    "keywords": keywords,
                    "limit": limit,
                    "min_score": min_score
                }
            
            # Execute hybrid query
            results = self.neo4j_manager.execute_query(query, parameters)
            
            # Post-process results to get parent chunks for non-chunk nodes
            enhanced_results = []
            chunk_cache = {}  # Cache to avoid duplicate chunk queries
            
            for result in results:
                node = result["node"]
                node_labels = result.get("node_labels", [])
                node_type = node_labels[0] if node_labels else "Unknown"
                
                # If it's already a Chunk, use it directly
                if node_type == "Chunk":
                    result_item = {
                        "node_id": node.get("id", "unknown"),
                        "node_type": node_type,
                        "content": node.get("text", ""),
                        "relevance_score": float(result["combined_score"]),
                        "semantic_score": float(result["semantic_score"]),
                        "keyword_score": float(result["keyword_score"]),
                        "type_bonus": float(result["type_bonus"]),
                        "content_bonus": float(result["content_bonus"]),
                        "search_method": f"hybrid_search{search_method_suffix}",
                        "original_node_type": node_type
                    }
                    
                    # Add description if available
                    description = node.get("description", "")
                    if description:
                        result_item["description"] = description
                    
                    enhanced_results.append(result_item)
                
                # If it's Entity, Table, Item, etc., find its parent Chunk
                elif node_type in ["Entity", "Table", "Item", "Paragraph", "Section"]:
                    node_id = node.get("id", "unknown")
                    
                    # Check cache first
                    if node_id in chunk_cache:
                        parent_chunks = chunk_cache[node_id]
                    else:
                        # Query to find parent chunks
                        parent_results = self.neo4j_manager.execute_query(
                            PARENT_CONTENT_QUERY, 
                            {"node_id": node_id}
                        )
                        chunk_cache[node_id] = parent_results
                        parent_chunks = parent_results
                    
                    # Add parent chunks with inherited score
                    for parent_result in parent_chunks:
                        chunk_node = parent_result["chunk"]
                        chunk_id = parent_result["chunk_id"]
                        
                        # Check if this chunk is already in results to avoid duplicates
                        if not any(r["node_id"] == chunk_id for r in enhanced_results):
                            result_item = {
                                "node_id": chunk_id,
                                "node_type": "Chunk",
                                "content": parent_result["chunk_text"] or "",
                                "relevance_score": float(result["combined_score"]) * 0.9,  # Slightly reduce score for inherited
                                "semantic_score": float(result["semantic_score"]) * 0.9,
                                "keyword_score": float(result["keyword_score"]),
                                "type_bonus": 0.05,  # Chunk bonus
                                "content_bonus": float(result["content_bonus"]),
                                "search_method": f"hybrid_search_via_child{search_method_suffix}",
                                "original_node_type": node_type,
                                "original_node_id": node_id,
                                "inherited_from": f"{node_type}:{node_id}"
                            }
                            
                            # Add description if available
                            description = chunk_node.get("description", "")
                            if description:
                                result_item["description"] = description
                            
                            enhanced_results.append(result_item)
                
                # For other node types, try to find related chunks
                else:
                    # Try to find any related chunks through relationships
                    related_results = self.neo4j_manager.execute_query(
                        RELATED_CONTENT_QUERY, 
                        {"node_id": node.get("id", "unknown")}
                    )
                    
                    for related_result in related_results:
                        chunk_node = related_result["chunk"]
                        chunk_id = related_result["chunk_id"]
                        
                        if not any(r["node_id"] == chunk_id for r in enhanced_results):
                            result_item = {
                                "node_id": chunk_id,
                                "node_type": "Chunk", 
                                "content": related_result["chunk_text"] or "",
                                "relevance_score": float(result["combined_score"]) * 0.8,  # Further reduce for distant relation
                                "semantic_score": float(result["semantic_score"]) * 0.8,
                                "keyword_score": float(result["keyword_score"]),
                                "type_bonus": 0.05,
                                "content_bonus": float(result["content_bonus"]),
                                "search_method": f"hybrid_search_via_relation{search_method_suffix}",
                                "original_node_type": node_type,
                                "original_node_id": node.get("id", "unknown"),
                                "inherited_from": f"{node_type}:{node.get('id', 'unknown')}"
                            }
                            
                            enhanced_results.append(result_item)
            
            # Sort by relevance score and remove duplicates
            seen_chunks = set()
            final_results = []
            for result in sorted(enhanced_results, key=lambda x: x["relevance_score"], reverse=True):
                if result["node_id"] not in seen_chunks:
                    seen_chunks.add(result["node_id"])
                    final_results.append(result)
            
            # Limit final results
            final_results = final_results[:limit]
            
            self.logger.info(f"Enhanced hybrid search found {len(final_results)} chunk-based results")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            # Fallback to basic semantic search with same embedding
            return self.search_knowledge_graph(question_embedding, limit)

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text for keyword matching (multi-language support)
        
        Args:
            text: Input text in any language
            
        Returns:
            List of keywords
        """
        import re
        
        # Multi-language stop words
        stop_words = {
            # English
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'what', 'how', 'when', 'where', 'why', 'which', 'who', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            # Vietnamese
            'là', 'của', 'có', 'được', 'một', 'các', 'trong', 'cho', 'với', 'về', 'từ', 'khi',
            'như', 'để', 'hay', 'này', 'đó', 'sẽ', 'đã', 'và', 'hoặc', 'nhưng', 'nếu', 'thì',
            # Japanese common particles and words
            'は', 'が', 'を', 'に', 'で', 'と', 'の', 'も', 'から', 'まで', 'より', 'へ', 'や',
            'です', 'である', 'だ', 'であり', 'として', 'について', 'において', 'による',
            # Chinese common words
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看'
        }
        
        # Universal Unicode word pattern that works with all languages
        # Includes Latin, Cyrillic, CJK, Arabic, Hebrew, Thai, etc.
        unicode_word_pattern = r'[\w\u00C0-\u024F\u1E00-\u1EFF\u0100-\u017F\u0180-\u024F\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F\u1C80-\u1C8F\u4E00-\u9FFF\u3400-\u4DBF\u20000-\u2A6DF\u2A700-\u2B73F\u2B740-\u2B81F\u2B820-\u2CEAF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\uD7B0-\uD7FF\u0590-\u05FF\uFB1D-\uFB4F\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\u0E00-\u0E7F\u1000-\u109F\u1780-\u17FF\u1800-\u18AF]+'
        
        # Extract words using Unicode-aware pattern
        words = re.findall(unicode_word_pattern, text.lower(), re.UNICODE)
        
        if not words:
            return []
        
        # Filter words based on language-specific rules
        filtered_keywords = []
        for word in words:
            # Skip if too short (but allow single CJK characters)
            min_length = 1 if self._is_cjk_word(word) else 2
            if len(word) < min_length:
                continue
                
            # Skip if too long (likely not a meaningful keyword)
            if len(word) > 20:
                continue
                
            # Skip if it's just numbers
            if word.isdigit():
                continue
                
            # Skip stop words
            if word in stop_words:
                continue
                
            # Skip if mostly punctuation
            if not self._is_meaningful_word(word):
                continue
                
            filtered_keywords.append(word)
        
        # Return unique keywords, limited to most important ones (same as original)
        return list(set(filtered_keywords))[:10]
    
    def _is_cjk_word(self, word: str) -> bool:
        """Check if word contains CJK (Chinese, Japanese, Korean) characters"""
        cjk_ranges = [
            (0x4E00, 0x9FFF),   # CJK Unified Ideographs
            (0x3400, 0x4DBF),   # CJK Extension A
            (0x3040, 0x309F),   # Hiragana
            (0x30A0, 0x30FF),   # Katakana
            (0xAC00, 0xD7AF),   # Hangul
        ]
        
        return any(
            any(start <= ord(char) <= end for start, end in cjk_ranges)
            for char in word
        )
    
    def _is_meaningful_word(self, word: str) -> bool:
        """Check if word is meaningful (contains enough letters)"""
        # Count letters vs other characters
        letter_count = sum(1 for char in word if char.isalpha() or ord(char) > 127)  # Include Unicode letters
        
        # At least 60% should be letters for short words, 40% for longer words
        min_letter_ratio = 0.6 if len(word) <= 5 else 0.4
        
        return letter_count / len(word) >= min_letter_ratio

    def search_knowledge_graph(self, 
                              question_embedding: list,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search Neo4j knowledge graph for relevant information using cosine similarity
        
        Args:
            question_embedding: Pre-computed embedding for the question (required)
            limit: Maximum number of results to return
            
        Returns:
            List of relevant nodes with metadata
        """
        try:
            # Validate that embedding is provided
            if not question_embedding:
                raise ValueError("question_embedding is required and cannot be empty")
            
            self.logger.info("Using pre-computed embedding for semantic search")
            
            # Use embedding directly
            embedding = question_embedding
            
            # Choose query based on scope
            if self.document_name:
                # Document-scoped search
                query = SEMANTIC_SEARCH_SCOPED_QUERY
                search_method_suffix = "_document_scoped"
                parameters = {
                    "embedding": embedding,
                    "limit": limit,
                    "document_name": self.document_name
                }
            else:
                # Global search - entire graph
                query = SEMANTIC_SEARCH_GLOBAL_QUERY
                search_method_suffix = "_global_scoped"
                parameters = {
                    "embedding": embedding,
                    "limit": limit
                }
            
            results = self.neo4j_manager.execute_query(query, parameters)
            
            # Format results consistently
            formatted_results = []
            for result in results:
                node = result["node"]
                node_labels = result.get("node_labels", [])
                node_type = node_labels[0] if node_labels else "Unknown"
                
                # Get content from various fields
                content = result.get("content") or result.get("name") or result.get("summary") or result.get("description") or ""
                
                result_item = {
                    "node_id": node.get("id", "unknown"),
                    "node_type": node_type,
                    "content": content,
                    "relevance_score": float(result["similarity"]),
                    "search_method": f"cosine_similarity{search_method_suffix}",
                    "original_node_type": node_type
                }
                
                # Add additional metadata if available
                if result.get("description"):
                    result_item["description"] = result["description"]
                if result.get("name"):
                    result_item["name"] = result["name"]
                if result.get("summary"):
                    result_item["summary"] = result["summary"]
                
                formatted_results.append(result_item)
            
            self.logger.info(f"Found {len(formatted_results)} results using cosine similarity search")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []

    def build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Build context string from search results"""
        if not search_results:
            return ""
        
        context_parts = ["=== CONTEXT INFORMATION ===", ""]
        
        # Group by node type
        chunks = [r for r in search_results if r["node_type"] == "Chunk"]
        
        # Add chunk information (highest priority)
        context_parts.extend(["## DOCUMENT SECTIONS:", ""])
        for chunk in chunks:
            content = chunk.get("content", "")
            
            if content:
                context_parts.append(f"Content: {content}")
            context_parts.append("\n")
        
        return "\n".join(context_parts)

    def close_connection(self):
        """Close Neo4j connection"""
        try:
            if self.neo4j_manager:
                # Use disconnect instead of close like original
                self.neo4j_manager.disconnect()
            self.logger.info("GraphRetriever closed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
