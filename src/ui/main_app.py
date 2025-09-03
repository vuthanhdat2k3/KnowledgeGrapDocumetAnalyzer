"""
Knowledge Graph Document Analyzer - Main Streamlit App
Direct Pipeline Architecture - Personal Project
"""
import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import pytz
import traceback

# Add src to path - consistent with run_app.py pattern
current_file = Path(__file__)
project_root = current_file.parent.parent.parent  # Go up from src/ui/ to project root
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Debug info (will show in streamlit logs)
print(f"🐍 Streamlit app starting...")
print(f"📂 Current file: {current_file}")
print(f"📂 Project root: {project_root}")
print(f"📂 Src path: {src_path}")
print(f"🔍 Working directory: {os.getcwd()}")

try:
    # Import configurations
    from config.app_config import app_config
    from config.ai_config import ai_config
    
    # Import core modules
    from src.core.knowledge_graph.neo4j_manager import neo4j_manager
    from src.core.knowledge_graph.kg_operations import kg_operations
    from src.core.pipeline.document_processor import DocumentProcessor
    from src.core.profile_generation.profile_generator import ProfileGenerator
    from src.core.qa_generation.question_evaluator import QuestionEvaluator
    
    print("✅ All imports successful")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    st.error(f"❌ Import Error: {e}")
    st.error(f"Working directory: {os.getcwd()}")
    st.error(f"Project root: {project_root}")
    st.error(f"Files in project root: {list(project_root.glob('*')) if project_root.exists() else 'Project root not found'}")
    st.stop()

# Page config
st.set_page_config(
    page_title=app_config.name,
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

def initialize_system():
    """Initialize system components"""
    try:
        # Connect to Neo4j
        if not neo4j_manager.connect():
            st.error("❌ Không thể kết nối với Neo4j!")
            return False
        
        # Initialize DocumentProcessor for Direct Pipeline
        if ai_config.api_key:
            processor = DocumentProcessor()
            st.session_state['document_processor'] = processor
            
            # Initialize ProfileGenerator
            profile_generator = ProfileGenerator()
            st.session_state['profile_generator'] = profile_generator
            
            # Initialize QuestionEvaluator
            question_evaluator = QuestionEvaluator()
            st.session_state['question_evaluator'] = question_evaluator
        
        # Initialize processing history
        if 'processing_history' not in st.session_state:
            st.session_state['processing_history'] = []
        
        return True
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo: {e}")
        return False

def show_system_status():
    """Display system status"""
    st.header("📊 Trạng thái hệ thống")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗄️ Neo4j Database")
        try:
            health = neo4j_manager.health_check()
            if health["status"] == "connected":
                st.markdown('<div class="status-box status-success">✅ Kết nối thành công</div>', unsafe_allow_html=True)
                if "stats" in health:
                    stats = health["stats"]
                    st.metric("Entities", stats.get("nodes", 0))
                    st.metric("Relationships", stats.get("relationships", 0))
            else:
                st.markdown('<div class="status-box status-error">❌ Kết nối thất bại</div>', unsafe_allow_html=True)
                st.write(health.get("message", "Unknown error"))
        except Exception as e:
            st.markdown('<div class="status-box status-error">❌ Lỗi kết nối</div>', unsafe_allow_html=True)
            st.write(str(e))
    
    with col2:
        st.subheader("🧠 Direct Pipeline Engine")
        try:
            # Check DocumentProcessor status
            has_api_key = bool(ai_config.api_key)
            has_processor = 'document_processor' in st.session_state
            
            if has_processor and has_api_key:
                st.markdown('<div class="status-box status-success">✅ Direct Pipeline Ready</div>', unsafe_allow_html=True)
                st.write("**Status**: DocumentProcessor initialized")
                st.write("**OpenAI API**: ✅ Configured")
                st.write("**Components**: File extractors, Neo4j integration")
            elif has_api_key:
                st.markdown('<div class="status-box status-warning">⚡ Pipeline Available - Needs Initialization</div>', unsafe_allow_html=True) 
                st.write("**OpenAI API**: ✅ Configured")
                st.write("**Status**: Ready to initialize")
            else:
                st.markdown('<div class="status-box status-error">❌ Missing OpenAI API Key</div>', unsafe_allow_html=True)
                st.write("**Status**: Configure API key to enable pipeline")
                
        except Exception as e:
            st.markdown('<div class="status-box status-error">❌ Pipeline Error</div>', unsafe_allow_html=True)
            st.write(str(e))

def show_architecture_page():
    """Show architecture overview"""
    st.header("🏗️ Direct Pipeline Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Completed Components")
        st.markdown("""
        **🏗️ Core Pipeline** (`src/core/pipeline/`)
        - [x] `document_processor.py` - Main orchestrator with 6-step pipeline
        - [x] File type detection and routing
        - [x] Integration with specialized extractors
        
        **🎯 Node Extractors** (`src/core/node_extractor/`)
        - [x] `base_node_extractor.py` - Abstract base class
        - [x] `docx_node_extractor.py` - DOCX-specific extraction
        - [x] `pdf_node_extractor.py` - PDF-specific extraction  
        - [x] `excel_node_extractor.py` - Excel-specific extraction
        - [x] Factory function for extractor selection
        
        **📝 Specialized Prompts** (`src/core/node_extractor/prompts/`)
        - [x] `docx_prompts.py` - DOCX extraction prompts
        - [x] `pdf_prompts.py` - PDF extraction prompts
        - [x] `excel_prompts.py` - Excel extraction prompts
        
        **🗄️ Knowledge Graph** (`src/core/knowledge_graph/`)
        - [x] `kg_operations.py` - Direct Neo4j operations
        - [x] `neo4j_manager.py` - Database management
        """)
    
    with col2:
        st.subheader("🔄 Implementation TODOs")
        st.markdown("""
        **🔗 Missing Components** (Need Implementation)
        - [ ] LLM client integration in base extractors
        - [ ] Image description components for each file type
        - [ ] Markdown conversion components  
        - [ ] Markdown chunking components
        - [ ] Complete end-to-end testing
        
        **🚀 Next Steps:**
        - [ ] Implement LLM client in `BaseNodeExtractor` 
        - [ ] Complete image describer components
        - [ ] Complete markdown converter components
        - [ ] Complete markdown chunker components
        - [ ] Test full pipeline with real documents
        
        **🏛️ Architecture Benefits:**
        - Clear separation of responsibilities
        - File-type specific optimization
        - No wrapper abstraction complexity
        - Easy to test and maintain
        - Scalable for new file types
        
        **🗂️ Legacy Files Moved to `legacy_files/`:**
        - `graphiti_wrapper.py` - Replaced by direct pipeline
        - `entity_extractor.py` - Replaced by node extractors
        """)


def show_document_processing_page():
    """Show document processing page with file upload and processing"""
    st.header("📁 Document Processing Pipeline")
    
    st.info("""
    **🚀 Direct Pipeline Features:**
    - **File Type Detection**: Tự động nhận diện DOCX, PDF, Excel
    - **Image Description**: Mô tả hình ảnh trong document bằng AI
    - **Markdown Conversion**: Chuyển đổi sang markdown format
    - **Smart Chunking**: Chia nhỏ document thành chunks có ý nghĩa
    - **Node Extraction**: Trích xuất entities và relationships bằng LLM
    - **Neo4j Integration**: Lưu trữ vào knowledge graph database
    """)
    
    # File upload section
    st.subheader("📤 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Chọn file để xử lý:",
        type=['doc', 'docx', 'pdf', 'xlsx', 'xls'],
        help="Hỗ trợ: DOC (sẽ tự động chuyển sang DOCX bằng LibreOffice), DOCX, PDF, Excel files. File sẽ được xử lý qua 6-step pipeline."
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Tên file": uploaded_file.name,
            "Loại file": uploaded_file.type,
            "Kích thước": f"{uploaded_file.size / 1024:.1f} KB"
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**📋 Thông tin file:**")
            for key, value in file_details.items():
                st.write(f"• {key}: {value}")
        
        with col2:
            st.write("**🔧 Trạng thái Pipeline:**")
            if 'document_processor' in st.session_state:
                st.success("✅ Pipeline sẵn sàng")
                st.write("• DocumentProcessor: ✅")
                st.write("• LLM Client: ✅")
                st.write("• Neo4j: ✅")
            else:
                st.error("❌ Pipeline chưa khởi tạo")
                st.write("• DocumentProcessor: ❌")
                st.write("• LLM Client: ❌")
                st.write("• Neo4j: ❌")
        
        # Process button
        if st.button("🚀 Bắt đầu xử lý Document", type="primary", disabled='document_processor' not in st.session_state):
            if 'document_processor' in st.session_state:
                process_uploaded_file(uploaded_file)
            else:
                st.error("❌ DocumentProcessor chưa được khởi tạo! Vui lòng kiểm tra OpenAI API key.")
    
    # Processing history
    if 'processing_history' in st.session_state and st.session_state['processing_history']:
        st.subheader("📚 Lịch sử xử lý")
        
        for i, history_item in enumerate(st.session_state['processing_history']):
            with st.expander(f"📄 {history_item.get('filename', 'Unknown')} - {history_item.get('timestamp', 'Unknown')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Trạng thái**: {history_item.get('status', 'Unknown')}")
                    st.write(f"**Loại file**: {history_item.get('file_type', 'Unknown')}")
                    st.write(f"**Chunks**: {history_item.get('chunks_processed', 0)}")
                with col2:
                    st.write(f"**Entities**: {history_item.get('entities_extracted', 0)}")
                    st.write(f"**Relationships**: {history_item.get('relationships_extracted', 0)}")
                    if history_item.get('neo4j_success'):
                        st.success("✅ Neo4j Import thành công")
                    else:
                        st.error("❌ Neo4j Import thất bại")
    
    # Pipeline status
    st.subheader("🔍 Pipeline Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Pipeline Components", "6/6", "✅ Complete")
        st.write("• File Detection")
        st.write("• Image Description")
        st.write("• Markdown Conversion")
    
    with col2:
        st.metric("File Types", "3/3", "✅ Supported")
        st.write("• DOCX")
        st.write("• PDF")
        st.write("• Excel")
    
    with col3:
        st.metric("Neo4j Integration", "2/3", "🔄 Partial")
        st.write("• DOCX: ✅")
        st.write("• Excel: ✅")
        st.write("• PDF: 🔄")

def process_uploaded_file(uploaded_file):
    """Process uploaded file using the document processor pipeline"""
    try:
        # Create temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        st.info(f"📁 File tạm thời được lưu tại: {tmp_file_path}")
        
        # Process with pipeline
        with st.spinner("🔄 Đang xử lý document..."):
            processor = st.session_state['document_processor']
            result = processor.process_document(tmp_file_path)
        
        # Save to processing history
        history_item = {
            'filename': uploaded_file.name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': result.get('status', 'unknown'),
            'file_type': result.get('file_type', 'unknown'),
            'chunks_processed': result.get('chunks_processed', 0),
            'entities_extracted': result.get('entities_extracted', 0),
            'relationships_extracted': result.get('relationships_extracted', 0),
            'neo4j_success': result.get('neo4j_result', {}).get('success', False)
        }
        
        if 'processing_history' not in st.session_state:
            st.session_state['processing_history'] = []
        st.session_state['processing_history'].insert(0, history_item)
        
        # Display results
        st.subheader("📋 Kết quả xử lý")
        
        if result.get('status') == 'success':
            st.success("✅ Xử lý thành công!")
            
            # Results summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chunks xử lý", result.get('chunks_processed', 0))
            with col2:
                st.metric("Entities", result.get('entities_extracted', 0))
            with col3:
                st.metric("Relationships", result.get('relationships_extracted', 0))
            
            # Pipeline steps summary
            st.subheader("🔍 Chi tiết Pipeline Steps")
            steps = [
                ("1️⃣ File Detection", "✅", f"Detected: {result.get('file_type', 'Unknown')}"),
                ("2️⃣ Image Description", "✅", "Completed"),
                ("3️⃣ Markdown Conversion", "✅", "Completed"),
                ("4️⃣ Chunking", "✅", f"{result.get('chunks_processed', 0)} chunks created"),
                ("5️⃣ Node Extraction", "✅", f"{result.get('entities_extracted', 0)} entities, {result.get('relationships_extracted', 0)} relationships"),
                ("6️⃣ Neo4j Storage", "✅" if result.get('neo4j_result', {}).get('success') else "❌", 
                 "Success" if result.get('neo4j_result', {}).get('success') else "Failed")
            ]
            
            for step, status, details in steps:
                st.write(f"{step} {status} {details}")
            
            # Detailed results
            with st.expander("📊 Chi tiết kết quả JSON"):
                st.json(result)
            
            # Neo4j results
            neo4j_result = result.get('neo4j_result', {})
            if neo4j_result:
                st.subheader("🗄️ Neo4j Import Results")
                if neo4j_result.get('success'):
                    st.success(f"✅ Import thành công vào Neo4j")
                    st.write(f"**Document**: {neo4j_result.get('document_name', 'N/A')}")
                    st.write(f"**Builder**: {neo4j_result.get('builder_used', 'N/A')}")
                    st.write(f"**Method**: {neo4j_result.get('import_method', 'N/A')}")
                else:
                    st.error(f"❌ Import thất bại: {neo4j_result.get('error', 'Unknown error')}")
            
            # Graph file info
            if result.get('graph_file_path'):
                st.info(f"📄 Graph results được lưu tại: {result.get('graph_file_path')}")
        
        else:
            st.error(f"❌ Xử lý thất bại: {result.get('error', 'Unknown error')}")
            with st.expander("🔍 Chi tiết lỗi"):
                st.json(result)
        
        # Cleanup temporary file
        try:
            os.unlink(tmp_file_path)
            st.info("🧹 File tạm thời đã được xóa")
        except:
            pass
            
    except Exception as e:
        st.error(f"❌ Lỗi xử lý file: {e}")
        st.error(f"Chi tiết lỗi: {traceback.format_exc()}")
        
        # Save error to history
        if 'processing_history' not in st.session_state:
            st.session_state['processing_history'] = []
        
        error_history = {
            'filename': uploaded_file.name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'error',
            'error': str(e)
        }
        st.session_state['processing_history'].insert(0, error_history)


def show_demo_page():
    """Show demo page for testing"""
    st.header("🧪 Direct Pipeline Demo")
    
    st.info("""
    **🏗️ Architecture Overview:**
    1. **File Type Detection** → **Image Description**
    2. **Markdown Conversion** → **Chunking** 
    3. **Specialized Node Extraction** (DOCX/PDF/Excel)
    4. **Neo4j Knowledge Graph Storage**
    
    **💡 Note**: Để xử lý document thực tế, hãy sử dụng trang **"📁 Document Processing"**
    """)
    
    # Sample text input for testing
    st.subheader("🧪 Test với Sample Text")
    sample_text = st.text_area(
        "Test với sample text (Direct Pipeline Demo):",
        value="We are developing a mobile banking application using React Native and Node.js. The app will have authentication, payment features, and a dashboard for users.",
        height=150
    )

    if st.button("🚀 Test Direct Pipeline", type="primary"):
        if sample_text.strip():
            with st.spinner("Testing Direct Pipeline..."):
                try:
                    # Test with direct pipeline if available
                    if 'document_processor' in st.session_state:
                        processor = st.session_state['document_processor']
                        
                        st.info("ℹ️ Real pipeline processes DOCX/PDF/Excel files. This is a simplified test.")
                        
                        # Simulate processing result
                        result = {
                            "status": "demo_mode",
                            "message": "Direct Pipeline Ready - Full implementation requires document files",
                            "pipeline_components": [
                                "✅ DocumentProcessor initialized",
                                "✅ File-specific node extractors (DOCX, PDF, Excel)",
                                "✅ Specialized prompts for each file type", 
                                "✅ Neo4j integration via kg_operations",
                                "✅ Factory pattern for extractor selection"
                            ],
                            "next_steps": "Upload DOCX/PDF/Excel files for full processing"
                        }
                        
                    else:
                        result = {
                            "status": "not_initialized",
                            "message": "DocumentProcessor not initialized - check OpenAI API key"
                        }
                    
                    st.subheader("📋 Direct Pipeline Status")
                    st.json(result)
                        
                except Exception as e:
                    st.error(f"❌ Error testing pipeline: {e}")
        else:
            st.warning("⚠️ Please enter some text!")

def get_available_documents():
    """Get list of available documents from Neo4j database"""
    try:
        # Query to get all document names
        query = """
        MATCH (d:Document) 
        WHERE d.file_name IS NOT NULL 
        RETURN DISTINCT d.file_name AS document_name 
        ORDER BY d.file_name;
        """
        
        results = neo4j_manager.execute_query(query=query)
        
        document_names = []
        for record in results:
            doc_name = record.get("document_name")
            if doc_name and doc_name.strip():
                document_names.append(doc_name.strip())
        
        return document_names
        
    except Exception as e:
        st.error(f"❌ Lỗi lấy danh sách documents: {e}")
        return []

def get_document_id_from_name(document_name: str) -> Optional[str]:
    """Get document ID from document name"""
    try:
        # Query to get document ID by file_name
        query = """
        MATCH (d:Document {file_name: $document_name})
        RETURN d.id as document_id
        LIMIT 1
        """
        
        results = neo4j_manager.execute_query(query=query, parameters={"document_name": document_name})
        
        if results and len(results) > 0:
            document_id = results[0].get("document_id")
            if document_id:
                return document_id
        
        return None
        
    except Exception as e:
        st.error(f"❌ Lỗi lấy Document ID: {e}")
        return None

def get_document_name_from_id(document_id: str) -> Optional[str]:
    """Get document name from document ID"""
    try:
        # Query to get document name by ID
        query = """
        MATCH (d:Document {id: $document_id})
        RETURN d.file_name as document_name
        LIMIT 1
        """
        
        results = neo4j_manager.execute_query(query=query, parameters={"document_id": document_id})
        
        if results and len(results) > 0:
            document_name = results[0].get("document_name")
            if document_name:
                return document_name
        
        return None
        
    except Exception as e:
        st.error(f"❌ Lỗi lấy Document Name: {e}")
        return None

def show_profile_generation_page():
    """Show profile generation page"""
    st.header("👤 Profile Generation")
    
    st.info("""
    **🎯 Profile Generation Features:**
    - **Document Analysis**: Phân tích nội dung document để trích xuất thông tin project
    - **Category Classification**: Tự động phân loại project theo các categories
    - **Project Description**: Tạo mô tả project comprehensive
    - **Knowledge Graph Integration**: Sử dụng Neo4j knowledge graph để retrieve context
    """)
    
    # Check if profile generator is available
    if 'profile_generator' not in st.session_state:
        st.error("❌ ProfileGenerator chưa được khởi tạo! Vui lòng kiểm tra OpenAI API key.")
        return
    
    # Document selection
    st.subheader("📋 Document Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get available documents from Neo4j
        available_documents = get_available_documents()
        
        # Create options list with "All Documents" as first option
        document_options = ["[Tất cả documents]"] + available_documents
        
        selected_document = st.selectbox(
            "Document Name:",
            options=document_options,
            index=0,
            help="Chọn document để phân tích profile"
        )
        
        # Convert selection to document_name
        document_name = None if selected_document == "[Tất cả documents]" else selected_document
        
        # Show document scope info
        if document_name:
            st.info(f"📄 **Document selected**: {document_name}")
        else:
            st.warning("⚠️ Vui lòng chọn một document cụ thể để generate profile!")
    
    with col2:
        st.write("**📊 Available Categories:**")
        st.write("• BusinessCategory: B2B, B2C, B2E, B2G, C2C")
        st.write("• BusinessSize: Small/Medium, Enterprise")
        st.write("• ServiceCategory: Finance, Education, Healthcare, etc.")
        st.write("• IndustryCategory: IT, Construction, Manufacturing, etc.")
        st.write("• ServiceType: Consulting, Development, MVP, etc.")
        
        # Show available documents count
        if available_documents:
            st.success(f"✅ {len(available_documents)} documents found")
            
            # Show first few document names
            with st.expander("📋 Document List"):
                for i, doc_name in enumerate(available_documents[:5], 1):
                    st.write(f"{i}. {doc_name}")
                if len(available_documents) > 5:
                    st.write(f"... và {len(available_documents) - 5} documents khác")
        else:
            st.warning("⚠️ Không tìm thấy documents trong Neo4j")
    
    # Generate profile button
    if st.button("🚀 Generate Profile", type="primary", disabled=not document_name):
        if document_name:
            # Get document ID from document name
            document_id = get_document_id_from_name(document_name)
            if document_id:
                generate_profile(document_id, top_k=5)  # Use default top_k=5
            else:
                st.error(f"❌ Không thể tìm thấy Document ID cho: {document_id}")
        else:
            st.warning("⚠️ Vui lòng chọn một document!")
    
    # Profile generation history
    if 'profile_history' in st.session_state and st.session_state['profile_history']:
        st.subheader("📚 Profile Generation History")
        
        for i, history_item in enumerate(st.session_state['profile_history']):
            # Get document name for display
            document_id = history_item.get('document_id', 'Unknown')
            document_name = get_document_name_from_id(document_id) if document_id != 'Unknown' else 'Unknown'
            display_name = document_name if document_name else f"ID: {document_id}"
            
            with st.expander(f"📄 {display_name} - {history_item.get('timestamp', 'Unknown')}"):
                
                if history_item.get('status') == 'success':
                    # Show categories
                    st.write("**🏷️ Generated Categories:**")
                    categories = history_item.get('categories', {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        for category, info in list(categories.items())[:3]:
                            st.write(f"**{category}**: {info.get('label', 'Unknown')}")
                    
                    with col2:
                        for category, info in list(categories.items())[3:]:
                            st.write(f"**{category}**: {info.get('label', 'Unknown')}")
                    
                    # Show project description
                    st.write("**📝 Project Description:**")
                    description = history_item.get('project_description', 'N/A')
                    st.write(description)
                    
                    # Show detailed results
                    with st.expander("📊 Chi tiết Categories"):
                        st.json(categories)
                
                else:
                    st.error(f"❌ Generation failed: {history_item.get('error', 'Unknown error')}")

def generate_profile(document_id: str, top_k: int):
    """Generate profile for given document ID"""
    try:
        
        with st.spinner("🔄 Đang generate profile..."):
            profile_generator = st.session_state['profile_generator']
            result = profile_generator.run(document_id, top_k=top_k)
        
        # Save to profile history
        history_item = {
            'document_id': document_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'success',
            'top_k': top_k,
            'categories': result.get('categories', {}),
            'project_description': result.get('project_description', ''),
        }
        
        if 'profile_history' not in st.session_state:
            st.session_state['profile_history'] = []
        st.session_state['profile_history'].insert(0, history_item)
        
        # Display results
        st.subheader("📋 Profile Generation Results")
        
        st.success("✅ Profile generation thành công!")
        
        # Show categories in a nice format
        st.subheader("🏷️ Project Categories")
        categories = result.get('categories', {})
        
        if categories:
            col1, col2 = st.columns(2)
            
            category_items = list(categories.items())
            mid_point = len(category_items) // 2
            
            with col1:
                for category, info in category_items[:mid_point]:
                    with st.container():
                        st.write(f"**{category}**")
                        st.info(f"🎯 {info.get('label', 'Unknown')}")
            
            with col2:
                for category, info in category_items[mid_point:]:
                    with st.container():
                        st.write(f"**{category}**")
                        st.info(f"🎯 {info.get('label', 'Unknown')}")
        
        # Show project description
        st.subheader("📝 Project Description")
        description = result.get('project_description', '')
        if description:
            st.write(description)
        else:
            st.warning("⚠️ Không thể tạo project description")
        
        # Detailed results
        with st.expander("📊 Detailed Categories Results"):
            st.json(categories)
        
        # Summary metrics
        st.subheader("📈 Generation Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Categories Generated", len(categories))
        with col2:
            st.metric("Top K Used", top_k)
        with col3:
            successful_categories = sum(1 for info in categories.values() if info.get('label') != 'Unknown')
            st.metric("Successful Classifications", successful_categories)
            
    except Exception as e:
        st.error(f"❌ Lỗi generate profile: {e}")
        st.error(f"Chi tiết lỗi: {traceback.format_exc()}")
        
        # Save error to history
        if 'profile_history' not in st.session_state:
            st.session_state['profile_history'] = []
        
        error_history = {
            'document_id': document_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'error',
            'error': str(e)
        }
        st.session_state['profile_history'].insert(0, error_history)

def show_question_evaluation_page():
    """Show question evaluation page"""
    st.header("❓ Question Evaluation")
    
    st.info("""
    **🤖 Question Evaluation Features:**
    - **Question Answering**: Trả lời câu hỏi dựa trên knowledge graph
    - **Hybrid Search**: Sử dụng hybrid search để tìm relevant context
    - **Batch Processing**: Xử lý nhiều câu hỏi từ JSON file
    - **Multi-language Support**: Hỗ trợ tiếng Việt và tiếng Anh
    """)
    
    # Check if question evaluator is available
    if 'question_evaluator' not in st.session_state:
        st.error("❌ QuestionEvaluator chưa được khởi tạo! Vui lòng kiểm tra OpenAI API key.")
        return
    
    # Show question section directly
    show_question_section()

def show_question_section():
    """Show simplified question answering section"""
    st.subheader("❓ Question Answering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get available documents from Neo4j
        available_documents = get_available_documents()
        
        # Create options list with "All Documents" as first option
        document_options = ["[Tất cả documents]"] + available_documents
        
        selected_document = st.selectbox(
            "Document Name:",
            options=document_options,
            index=0,
            help="Chọn document cụ thể hoặc search toàn bộ knowledge graph."
        )
        
        # Convert selection to document_name
        document_name = None if selected_document == "[Tất cả documents]" else selected_document
        
        # Show document scope info
        if document_name:
            st.info(f"📄 **Document scope**: {document_name}")
        else:
            st.info("🌐 **Document scope**: Toàn bộ knowledge graph")
    
    with col2:
        st.write("**📄 Available Documents:**")
        
        # Show available documents count
        if available_documents:
            st.success(f"✅ {len(available_documents)} documents found")
            
            # Show first few document names
            with st.expander("📋 Document List"):
                for i, doc_name in enumerate(available_documents[:5], 1):
                    st.write(f"{i}. {doc_name}")
                if len(available_documents) > 5:
                    st.write(f"... và {len(available_documents) - 5} documents khác")
        else:
            st.warning("⚠️ Không tìm thấy documents trong Neo4j")
    
    # Output filename
    st.subheader("⚙️ Settings")
    output_filename = st.text_input(
        "Output Filename:",
        value="question_answers.json",
        help="Tên file để lưu kết quả"
    )
    
    # Process all questions button
    if st.button("🚀 Process All Questions", type="primary"):
        process_all_questions_simplified(document_name, output_filename)
    
    # Display previous results if available
    if 'last_qa_results' in st.session_state:
        st.markdown("---")
        display_qa_results(st.session_state['last_qa_results'])

def process_all_questions_simplified(document_name: Optional[str], output_filename: str):
    """Process all questions using ask_all_questions method"""
    try:
        with st.spinner("🔄 Đang xử lý tất cả questions..."):
            question_evaluator = st.session_state['question_evaluator']
            
            # Update document scope if provided
            if document_name:
                question_evaluator.document_name = document_name
                question_evaluator.retriever.document_name = document_name
            else:
                # Reset to global scope
                question_evaluator.document_name = None
                question_evaluator.retriever.document_name = None
            
            # Process all questions using default model (gpt-4.1)
            results = question_evaluator.ask_all_questions(
                output_path=output_filename,
                model_name="gpt-4.1"
            )
        
        if results:
            st.success(f"✅ Đã xử lý {len(results)} câu hỏi thành công!")
            
            # Save results to session state to persist across reruns
            
            # Get Vietnam timezone
            vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
            vn_time = datetime.now(vn_tz)
            
            st.session_state['last_qa_results'] = {
                'results': results,
                'document_scope': document_name or 'global_graph',
                'output_file': output_filename,
                'timestamp': vn_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to QA history  
            answerable_count = sum(1 for r in results if r.get('answerable', False))
            history_item = {
                'type': 'batch_processing',
                'total_questions': len(results),
                'answerable_count': answerable_count,
                'timestamp': vn_time.strftime("%Y-%m-%d %H:%M:%S"),
                'document_scope': document_name or 'global_graph',
                'output_file': output_filename
            }
            
            if 'qa_history' not in st.session_state:
                st.session_state['qa_history'] = []
            st.session_state['qa_history'].insert(0, history_item)
            
            # Force display results immediately
            display_qa_results(st.session_state['last_qa_results'])
            
        else:
            st.error("❌ Không có câu hỏi nào được xử lý!")
            
    except Exception as e:
        st.error(f"❌ Lỗi xử lý questions: {e}")
        st.error(f"Chi tiết lỗi: {traceback.format_exc()}")

def display_qa_results(qa_results_data):
    """Display Q&A results with filtering options - persists across reruns"""
    results = qa_results_data['results']
    document_scope = qa_results_data['document_scope']
    output_file = qa_results_data['output_file']
    timestamp = qa_results_data['timestamp']
    
    # Show summary
    st.subheader("📊 Processing Summary")
    
    answerable_count = sum(1 for r in results if r.get('answerable', False))
    successful_count = sum(1 for r in results if not r.get('error'))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Questions", len(results))
    with col2:
        st.metric("Answerable", answerable_count)
    with col3:
        st.metric("Successful", successful_count)
    
    # Show processing info
    st.info(f"📄 **Document scope**: {document_scope} | ⏰ **Processed**: {timestamp}")
    
    # View options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📋 All Questions & Answers")
    with col2:
        show_details = st.checkbox("Show Details", value=True, help="Show related context and metadata", key="show_details_cb")
    
    # Filter options  
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        show_answerable = st.checkbox("Show Answerable Only", value=False, help="Filter to show only answerable questions", key="show_answerable_cb")
    with filter_col2:
        show_unanswerable = st.checkbox("Show Unanswerable Only", value=False, help="Filter to show only unanswerable questions", key="show_unanswerable_cb")
    
    # Apply filters
    filtered_results = results
    if show_answerable and not show_unanswerable:
        filtered_results = [r for r in results if r.get('answerable', False)]
    elif show_unanswerable and not show_answerable:
        filtered_results = [r for r in results if not r.get('answerable', False)]
    
    st.write(f"**Showing {len(filtered_results)} of {len(results)} questions**")
    
    # Display filtered results
    for i, result in enumerate(filtered_results):
        # Find original index for numbering
        original_index = results.index(result) + 1
        
        answerable_icon = "✅" if result.get('answerable') else "❌"
        question_preview = result.get('question', 'Unknown')[:80]
        
        with st.expander(f"{answerable_icon} Question {original_index}: {question_preview}..."):
            
            # Question
            st.write("**❓ Question:**")
            st.write(result.get('question', 'N/A'))
            
            # Answer status
            st.write(f"**🎯 Answerable**: {'✅ Yes' if result.get('answerable') else '❌ No'}")
            
            # Answer - show full content
            st.write("**💬 Answer:**")
            answer_text = result.get('answer', 'N/A')
            if answer_text and answer_text.strip():
                # Display full answer without truncation
                st.markdown(answer_text)
            else:
                st.write("*No answer provided*")
            
            # Show details if enabled
            if show_details:
                # Related context if available
                if result.get('related_context'):
                    st.write("**📚 Related Context:**")
                    st.markdown(result.get('related_context', ''))
    
    st.info(f"📁 Full results saved to: {output_file}")

def main():
    """Main app function"""
    # Header
    st.markdown('<h1 class="main-header">🕸️ Knowledge Graph Document Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Direct Pipeline Architecture - Personal Project</p>', unsafe_allow_html=True)
    
    # Initialize system
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = initialize_system()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Chọn trang:",
        ["📊 System Status", "🏗️ Architecture", "📁 Document Processing", "🧪 Demo", "👤 Profile Generation", "❓ Question Evaluation"]
    )
    
    # System info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Info")
    st.sidebar.write(f"**OpenAI API**: {'✅' if ai_config.api_key else '❌'}")
    st.sidebar.write(f"**Neo4j**: {'✅' if neo4j_manager.is_connected() else '❌'}")
    st.sidebar.write(f"**Pipeline**: {'✅' if 'document_processor' in st.session_state else '❌'}")
    st.sidebar.write(f"**Profile Gen**: {'✅' if 'profile_generator' in st.session_state else '❌'}")
    st.sidebar.write(f"**QA Evaluator**: {'✅' if 'question_evaluator' in st.session_state else '❌'}")
    
    # Pipeline status in sidebar
    if 'document_processor' in st.session_state:
        st.sidebar.markdown("### 🚀 Pipeline Status")
        st.sidebar.success("✅ DocumentProcessor Ready")
        st.sidebar.write("**File Types**: DOCX, PDF, Excel")
        st.sidebar.write("**Components**: 6/6 Ready")
        
        # Processing history summary
        if 'processing_history' in st.session_state and st.session_state['processing_history']:
            st.sidebar.markdown("### 📚 Recent Processing")
            recent_count = min(3, len(st.session_state['processing_history']))
            for i in range(recent_count):
                item = st.session_state['processing_history'][i]
                status_icon = "✅" if item.get('status') == 'success' else "❌"
                st.sidebar.write(f"{status_icon} {item.get('filename', 'Unknown')}")
        
        # New components status
        st.sidebar.markdown("### 🆕 New Components")
        if 'profile_generator' in st.session_state:
            st.sidebar.success("✅ ProfileGenerator Ready")
        if 'question_evaluator' in st.session_state:
            st.sidebar.success("✅ QuestionEvaluator Ready")
        
        # QA History summary
        if 'qa_history' in st.session_state and st.session_state['qa_history']:
            st.sidebar.markdown("### ❓ Recent Q&A")
            recent_qa_count = min(2, len(st.session_state['qa_history']))
            for i in range(recent_qa_count):
                qa_item = st.session_state['qa_history'][i]
                if qa_item.get('type') == 'batch_processing':
                    answerable_count = qa_item.get('answerable_count', 0)
                    total_questions = qa_item.get('total_questions', 0)
                    st.sidebar.write(f"📊 Batch: {answerable_count}/{total_questions} answered")
                else:
                    answerable_icon = "✅" if qa_item.get('answerable') else "❌"
                    question_preview = qa_item.get('question', 'Unknown')[:30]
                    st.sidebar.write(f"{answerable_icon} {question_preview}...")
        
        # Profile History summary
        if 'profile_history' in st.session_state and st.session_state['profile_history']:
            st.sidebar.markdown("### 👤 Recent Profiles")
            recent_profile_count = min(2, len(st.session_state['profile_history']))
            for i in range(recent_profile_count):
                profile_item = st.session_state['profile_history'][i]
                profile_icon = "✅" if profile_item.get('status') == 'success' else "❌"
                doc_id = profile_item.get('document_id', 'Unknown')
                doc_name = get_document_name_from_id(doc_id) if doc_id != 'Unknown' else 'Unknown'
                display_name = doc_name if doc_name else f"ID: {doc_id[:15]}..."
                st.sidebar.write(f"{profile_icon} {display_name}")
    else:
        st.sidebar.markdown("### 🚀 Pipeline Status")
        st.sidebar.error("❌ Pipeline Not Ready")
        st.sidebar.write("**Check**: OpenAI API Key")
        st.sidebar.write("**Status**: Needs Configuration")
    
    # Page routing
    if page == "📊 System Status":
        show_system_status()
    elif page == "🏗️ Architecture":
        show_architecture_page()
    elif page == "📁 Document Processing":
        show_document_processing_page()
    elif page == "🧪 Demo":
        show_demo_page()
    elif page == "👤 Profile Generation":
        show_profile_generation_page()
    elif page == "❓ Question Evaluation":
        show_question_evaluation_page()

if __name__ == "__main__":
    main()
