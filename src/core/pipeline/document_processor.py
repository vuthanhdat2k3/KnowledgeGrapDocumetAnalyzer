"""
Document Processing Pipeline
Direct pipeline với clear responsibilities cho mỗi component
"""
import os
import uuid
import logging
import asyncio
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json

# Import all components at module level for better performance
from ..llm_client import LLMClientFactory

# Image describers
from ..image_describer.docx_image_describer import DocxImageDescriber
from ..image_describer.pdf_image_describer import PdfImageDescriber
from ..image_describer.excel_image_describer import ExcelImageDescriber

# Markdown converters
from ..markdown_converter.docx_markdown_converter import DocxMarkdownConverter
from ..markdown_converter.pdf_markdown_converter import PdfMarkdownConverter
from ..markdown_converter.excel_markdown_converter import ExcelMarkdownConverter

# Markdown chunkers
from ..markdown_chunker.docx_markdown_chunker import DocxMarkdownChunker
from ..markdown_chunker.excel_markdown_chunker import ExcelMarkdownChunker

# Node extractors and builders
from ..node_extractor.docx_node_extractor import DocxNodeExtractor
from ..node_extractor.excel_node_extractor import ExcelNodeExtractor

# KG Builders
from ..kg_builders.docx_kg_builder import DocxKGBuilder
from ..kg_builders.excel_kg_builder import ExcelKGBuilder

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):        
        # Khởi tạo LLM Client
        self.llm_factory = LLMClientFactory()
        self.llm_client_gpt41_nano = self.llm_factory.get_client("gpt-4.1-nano")
        self.llm_client_gpt41 = self.llm_factory.get_client("gpt-4.1")
        

        # Khởi tạo describer
        self.docx_image_describer = DocxImageDescriber(self.llm_client_gpt41_nano)
        self.pdf_image_describer = PdfImageDescriber()
        self.excel_image_describer = ExcelImageDescriber(self.llm_client_gpt41_nano)

        # Khởi tạo converter
        self.docx_markdown_converter = DocxMarkdownConverter()
        self.pdf_markdown_converter = PdfMarkdownConverter()
        self.excel_markdown_converter = ExcelMarkdownConverter()
        
        # Khởi tạo chunker
        self.docx_markdown_chunker = DocxMarkdownChunker()
        self.excel_markdown_chunker = ExcelMarkdownChunker()

        # Khởi tạo node extractor
        self.docx_node_extractor = DocxNodeExtractor() 
        self.excel_node_extractor = ExcelNodeExtractor()

        # Khởi tạo KG Builders
        self.docx_kg_builder = DocxKGBuilder()
        self.excel_kg_builder = ExcelKGBuilder()

        logger.info("DocumentProcessor initialized")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Complete document processing pipeline
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dict: Processing result with statistics
        """
        try:
            logger.info(f"Starting document processing: {file_path}")
            
            # Step 1: Detect file type
            file_type = self._detect_file_type(file_path)
            logger.info(f"Detected file type: {file_type}")
            
            # Step 2: Image description (if needed)
            image_described_path = self._describe_images(file_path, file_type)
            logger.info(f"Image description completed: {image_described_path}")
            
            # Step 3: Convert to markdown
            markdown_path = self._convert_to_markdown(image_described_path, file_type)
            logger.info(f"Markdown conversion completed: {markdown_path}")
            
            # Step 4: Chunk markdown
            chunks = self._chunk_markdown(markdown_path, file_type)
            chunks_count = len(chunks)
            logger.info(f"Chunking completed: {chunks_count} chunks")
            
            # Step 5: Extract nodes with LLM and save graph results
            entities, relationships, graph_file_path = self._extract_nodes(chunks, file_type, file_path)
            logger.info(f"Node extraction completed: {len(entities)} entities, {len(relationships)} relationships")
            if graph_file_path:
                logger.info(f"Graph results saved to: {graph_file_path}")
            
            # Step 6: Save to Neo4j
            result = self._save_to_neo4j(entities, relationships, file_path, graph_file_path)
            logger.info(f"Neo4j storage completed: {result}")
            
            return {
                "status": "success",
                "file_path": file_path,
                "file_type": file_type,
                "chunks_processed": chunks_count,
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "graph_file_path": graph_file_path,
                "neo4j_result": result,
                "pipeline_complete": True
            }
            
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {e}")
            return {
                "status": "error",
                "error": str(e),
                "file_path": file_path,
                "pipeline_complete": False
            }
    
    def _detect_file_type(self, file_path: str) -> str:
        if file_path.endswith(('.docx', '.doc')):
            return 'docx'  # Xử lý cả .doc và .docx như nhau
        elif file_path.endswith('.pdf'):
            return 'pdf'
        elif file_path.endswith(('.xlsx', '.xls')):
            return 'excel'
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    
    def _describe_images(self, file_path: str, file_type: str) -> str:
        try:
            if file_type == 'docx':
                return self.docx_image_describer.run(file_path)
            elif file_type == 'pdf':
                return self.pdf_image_describer.run(file_path)
            elif file_type == 'excel':
                return self.excel_image_describer.run(file_path)
            else:
                logger.warning(f"No image describer for {file_type}, returning original path")
                return file_path
                
        except ImportError as e:
            logger.warning(f"Image describer not found for {file_type}: {e}")
            return file_path
        except Exception as e:
            logger.error(f"Error in image description for {file_type}: {e}")
            return file_path
    
    def _convert_to_markdown(self, file_path: str, file_type: str) -> str:
        if file_type == 'docx':
            return self.docx_markdown_converter.convert_to_markdown(file_path)
        elif file_type == 'pdf':
            return self.pdf_markdown_converter.convert_to_markdown(file_path)
        elif file_type == 'excel':
            # Excel converter trả về string content, cần save ra file
            markdown_content = self.excel_markdown_converter.convert_to_markdown(file_path)

            # Tạo output path cho markdown file
            input_path = Path(file_path)
            output_dir = input_path.parent
            markdown_filename = f"{input_path.stem}_converted.md"
            markdown_path = output_dir / markdown_filename
            
            # Save markdown content ra file
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Saved Excel markdown to: {markdown_path}")
            return str(markdown_path)
        else:
            raise ValueError(f"No markdown converter for file type: {file_type}")
    
    def _chunk_markdown(self, markdown_path: str, file_type: str):
        if file_type == 'docx' or file_type == 'pdf':
            chunk_dicts = self.docx_markdown_chunker.chunk_from_file(markdown_path)
            return chunk_dicts
        elif file_type == 'excel':
            # Excel chunker trả về List[Dict] với format đúng như extractor cần
            chunk_dicts = self.excel_markdown_chunker.chunk_from_file(markdown_path)
            logger.info(f"Excel chunker returned {len(chunk_dicts)} chunk dictionaries")
            return chunk_dicts
        else:
            raise ValueError(f"No chunker for file type: {file_type}")
    
    def _extract_nodes(self, chunks, file_type: str, source_file: str = None) -> Tuple[List[Dict], List[Dict], str]:
        """
        Extract nodes using specialized LLM extractors
        
        Args:
            chunks: For Excel: List[Dict] with content/metadata, For others: List[str] text chunks
            file_type: Type of original file
            source_file: Original source file path (for saving graph results)
            
        Returns:
            Tuple[List[Dict], List[Dict], str]: (entities, relationships, graph_file_path)
        """
        entities = []
        relationships = []
        
        try:
            if file_type == 'excel':                
                # Excel: Sử dụng batch processing với extract_graph
                logger.info(f"Using Excel batch processing for {len(chunks)} chunks")
                
                # Gọi extract_graph (async)
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Gọi extract_graph với đúng parameters như trong excel_node_extractor.py
                logger.info(f"Calling extract_graph with {len(chunks)} chunks (include_summary=True, normalize_graph=True)")
                graph_results = loop.run_until_complete(
                    self.excel_node_extractor.extract_graph(
                        chunks=chunks, 
                        include_summary=True, 
                        normalize_graph=True
                    )
                )
                
                logger.info(f"extract_graph returned {len(graph_results)} graph documents")
                
                if source_file:
                    # Tạo output path cho graph results file
                    input_path = Path(source_file)
                    output_dir = input_path.parent / "graph"
                    output_dir.mkdir(exist_ok=True)  # Tạo thư mục nếu chưa có
                    
                    graph_filename = f"{input_path.stem}_graph_results.json"
                    graph_file_path = output_dir / graph_filename
                    
                    # Lưu graph results ra file
                    with open(graph_file_path, 'w', encoding='utf-8') as f:
                        json.dump(graph_results, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Saved graph results to: {graph_file_path}")
                else:
                    graph_file_path = None
                    logger.warning("No source_file provided, skipping graph results file save")
                
                # Extract entities và relationships từ graph results (sau normalize_graph)
                for i, result in enumerate(graph_results):
                    nodes = result.get('nodes', [])
                    # Sau normalize_graph(), format sẽ có "edges" key chứ không phải "relationships"
                    edges_data = result.get('edges', [])
                    
                    logger.info(f"Processing result {i+1}/{len(graph_results)} with {len(nodes)} nodes, {len(edges_data)} edges")
                    
                    # Log additional data từ include_summary=True
                    if result.get('chunk_summary'):
                        logger.debug(f"Result {i+1} includes chunk summary: {len(result.get('chunk_summary', ''))} chars")
                    if result.get('chunk_embedding'):
                        logger.debug(f"Result {i+1} includes chunk embedding: {len(result.get('chunk_embedding', []))} dimensions")
                    
                    # Convert nodes thành entities
                    for node in nodes:
                        if node.get('type') != 'Description':  # Skip description nodes
                            entities.append({
                                'id': node.get('id'),
                                'text': node.get('name', ''),
                                'type': node.get('type', 'Unknown'),
                                'confidence': 1.0,
                                'metadata': node.get('properties', {}),
                                'relationships': []
                            })
                    
                    # Convert edges thành relationships format
                    for edge in edges_data:
                        relationships.append({
                            'source': edge.get('source'),
                            'target': edge.get('target'),
                            'type': edge.get('type'),
                            'properties': edge.get('properties', {})
                        })
                
                logger.info(f"Excel extraction completed: {len(entities)} entities, {len(relationships)} relationships")

            elif file_type == 'docx' or file_type == 'pdf':
                # DOCX: Sử dụng batch processing tương tự Excel
                logger.info(f"Using DOCX batch processing for {len(chunks)} chunks")
                
                # Gọi extract (async) từ DocxNodeExtractor
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Gọi extract method với chunks có format tương tự Excel
                logger.info(f"Calling extract with {len(chunks)} chunks")
                graph_results = loop.run_until_complete(
                    self.docx_node_extractor.extract(chunks)
                )
                
                logger.info(f"extract returned {len(graph_results)} graph documents")
                
                if source_file:
                    # Tạo output path cho graph results file
                    input_path = Path(source_file)
                    output_dir = input_path.parent / "graph"
                    output_dir.mkdir(exist_ok=True)  # Tạo thư mục nếu chưa có
                    
                    graph_filename = f"{input_path.stem}_graph_results.json"
                    graph_file_path = output_dir / graph_filename
                    
                    # Lưu graph results ra file
                    with open(graph_file_path, 'w', encoding='utf-8') as f:
                        json.dump(graph_results, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Saved graph results to: {graph_file_path}")
                else:
                    graph_file_path = None
                    logger.warning("No source_file provided, skipping graph results file save")
                
                # Extract entities và relationships từ graph results
                for i, result in enumerate(graph_results):
                    chunk_entities = result.get('entities', [])
                    chunk_relationships = result.get('relationships', [])
                    
                    logger.info(f"Processing result {i+1}/{len(graph_results)} with {len(chunk_entities)} entities, {len(chunk_relationships)} relationships")
                    
                    # Log additional data
                    if result.get('summary'):
                        logger.debug(f"Result {i+1} includes chunk summary: {len(result.get('summary', ''))} chars")
                    if result.get('embedding'):
                        logger.debug(f"Result {i+1} includes chunk embedding: {len(result.get('embedding', []))} dimensions")
                    
                    # Convert entities format
                    for entity in chunk_entities:
                        entities.append({
                            'id': entity.get('id'),
                            'name': entity.get('name'),
                            'type': entity.get('type', 'Unknown'),
                            'description_node': entity.get('description_node'),
                            'confidence': 1.0,
                            'metadata': {},
                            'relationships': []
                        })
                    
                    # Convert relationships format
                    for relationship in chunk_relationships:
                        relationships.append({
                            'id': relationship.get('id'),
                            'source': relationship.get('source'),
                            'target': relationship.get('target'),
                            'type': relationship.get('type'),
                            'description_node': relationship.get('description_node')
                        })
                
                logger.info(f"DOCX extraction completed: {len(entities)} entities, {len(relationships)} relationships")
            else:
                raise ValueError(f"Unsupported file type for node extraction: {file_type}")
            
            return entities, relationships, graph_file_path
            
        except ImportError as e:
            logger.error(f"Node extractor not found for {file_type}: {e}")
            return [], [], None
        except Exception as e:
            logger.error(f"Error in node extraction: {e}")
            return [], [], None
    
    def _save_to_neo4j(self, entities: List[Dict], relationships: List[Dict], source_file: str, graph_file_path: str = None) -> Dict:
        """
        Save extracted data to Neo4j using specialized KG builders
        
        Args:
            entities: List of entity dictionaries (from node extraction)
            relationships: List of relationship dictionaries (from node extraction)
            source_file: Original source file path
            graph_file_path: Path to saved graph results file (for file-based import)
            
        Returns:
            Dict: Save operation result
        """
        try:
            # 1. Detect file type
            file_type = self._detect_file_type(source_file)
            logger.info(f"Saving to Neo4j for file type: {file_type}")
            
            if file_type == 'excel':
                return self._save_excel_to_neo4j(entities, relationships, source_file, graph_file_path)
            elif file_type == 'docx' or file_type == 'pdf':
                return self._save_docx_to_neo4j(entities, relationships, source_file, graph_file_path)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_type}",
                    "entities_created": 0,
                    "relationships_created": 0
                }
            
        except Exception as e:
            logger.error(f"Error saving to Neo4j: {e}")
            return {
                "success": False,
                "error": str(e),
                "entities_created": 0,
                "relationships_created": 0
            }
    
    def _save_excel_to_neo4j(self, entities: List[Dict], relationships: List[Dict], source_file: str, graph_file_path: str = None) -> Dict:
        """
        Save Excel extracted data to Neo4j using ExcelKGBuilder
        """
        try:            
            # Get document name from file path
            document_name = Path(source_file).stem
            
            # Prioritize file-based import for better performance and data integrity
            if graph_file_path and Path(graph_file_path).exists():
                logger.info(f"Using file-based import from: {graph_file_path}")
                success = self.excel_kg_builder.build(str(graph_file_path), document_name)
                import_method = "file-based"
                
                if success:
                    return {
                        "success": True,
                        "document_name": document_name,
                        "entities_created": len(entities),
                        "relationships_created": len(relationships),
                        "builder_used": "ExcelKGBuilder",
                        "file_type": "excel",
                        "import_method": import_method,
                        "graph_file_path": graph_file_path
                    }
                else:
                    return {
                        "success": False,
                        "error": f"ExcelKGBuilder file-based import failed",
                        "entities_created": 0,
                        "relationships_created": 0,
                        "import_method": import_method,
                        "graph_file_path": graph_file_path
                    }
            else:
                # Graph file not available - this should not happen in normal flow
                logger.error("Graph file not available for Excel import. Excel requires graph results file.")
                return {
                    "success": False,
                    "error": "Graph file not available for Excel import",
                    "entities_created": 0,
                    "relationships_created": 0,
                    "import_method": "none",
                    "graph_file_path": graph_file_path
                }
                
        except Exception as e:
            logger.error(f"Error in Excel Neo4j save: {e}")
            return {
                "success": False,
                "error": str(e),
                "entities_created": 0,
                "relationships_created": 0
            }
    

    
    def _save_docx_to_neo4j(self, entities: List[Dict], relationships: List[Dict], source_file: str, graph_file_path: str = None) -> Dict:
        """
        Save DOCX extracted data to Neo4j using DocxKGBuilder
        """
        try:            
            # Get document name from file path
            document_name = Path(source_file).stem
            
            # Prioritize file-based import for better performance and data integrity
            if graph_file_path and Path(graph_file_path).exists():
                logger.info(f"Using file-based import from: {graph_file_path}")
                success = self.docx_kg_builder.build(str(graph_file_path), document_name)
                import_method = "file-based"
                
                if success:
                    return {
                        "success": True,
                        "document_name": document_name,
                        "entities_created": len(entities),
                        "relationships_created": len(relationships),
                        "builder_used": "DocxKGBuilder",
                        "file_type": "docx",
                        "import_method": import_method,
                        "graph_file_path": graph_file_path
                    }
                else:
                    return {
                        "success": False,
                        "error": f"DocxKGBuilder file-based import failed",
                        "entities_created": 0,
                        "relationships_created": 0,
                        "import_method": import_method,
                        "graph_file_path": graph_file_path
                    }
            else:
                # Graph file not available - this should not happen in normal flow
                logger.error("Graph file not available for DOCX import. DOCX requires graph results file.")
                return {
                    "success": False,
                    "error": "Graph file not available for DOCX import",
                    "entities_created": 0,
                    "relationships_created": 0,
                    "import_method": "none",
                    "graph_file_path": graph_file_path
                }
                
        except Exception as e:
            logger.error(f"Error in DOCX Neo4j save: {e}")
            return {
                "success": False,
                "error": str(e),
                "entities_created": 0,
                "relationships_created": 0
            }
