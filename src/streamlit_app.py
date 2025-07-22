import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface")

import streamlit as st
import os
import sys
from typing import List, Dict
import time
from datetime import datetime
import psutil

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Importing existing RAG components
try:
    from document_processor import DocumentProcessor
    from embedding_engine import EmbeddingEngine
    from vector_db_manager import VectorDBManager
    from llm_pipeline import LLMPipeline
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing RAG components: {str(e)}")
    COMPONENTS_AVAILABLE = False

class RAGChatbotUI:
    def __init__(self):
        self.doc_processor = None
        self.embedding_engine = None
        self.db_manager = None
        self.llm_pipeline = None
        self.session_initialized = False
        

    def initialize_components(self):
        """Initialize RAG components with local Docker Qdrant"""
        if not self.session_initialized and COMPONENTS_AVAILABLE:
            try:
                with st.spinner("Initializing RAG system components..."):
                    # Initialize components
                    self.doc_processor = DocumentProcessor()
                    self.embedding_engine = EmbeddingEngine()
                    
                    # Use local Docker Qdrant instead of cloud
                    self.db_manager = VectorDBManager(
                        url="http://localhost:6333",  # Local Docker
                        api_key=None  # No API key for local
                    )
                    self.llm_pipeline = LLMPipeline()
                    
                    if self.db_manager.test_connection():
                        st.session_state.rag_system = self
                        self.session_initialized = True
                        st.success("✅ RAG system initialized successfully!")
                    else:
                        st.error("❌ Failed to connect to vector database. Check your credentials.")
                        
            except Exception as e:
                st.error(f"❌ Error initializing components: {str(e)}")
                return False
        return True
    
    def process_query(self, query: str) -> Dict:
        """Process user query through the RAG pipeline"""
        start_time = time.time()
        
        try:
            # Load embedding model if needed
            if not hasattr(self.embedding_engine, 'model') or self.embedding_engine.model is None:
                with st.spinner("Loading embedding model..."):
                    self.embedding_engine.load_model()
            
            # Generate query embedding
            query_embedding = self.embedding_engine.embed_query(query)
            
            # Search vector database
            with st.spinner("Searching document database..."):
                search_results = self.db_manager.search_similar(query_embedding, top_k=5)
            
            if search_results:
                # Prepare context chunks
                context_chunks = []
                for result in search_results:
                    context_chunks.append({
                        'content': result.content,
                        'filename': result.filename,
                        'page_number': result.page_number,
                        'section_name': result.section_name,
                        'chunk_id': result.chunk_id
                    })
                
                # Generate answer
                with st.spinner("Generating answer..."):
                    answer_result = self.llm_pipeline.generate_answer_from_context(query, context_chunks)
                
                processing_time = time.time() - start_time
                
                return {
                    'success': True,
                    'answer_result': answer_result,
                    'processing_time': processing_time,
                    'chunks_found': len(search_results)
                }
            else:
                return {
                    'success': False,
                    'error': 'No relevant documents found in the database.',
                    'processing_time': time.time() - start_time
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

def display_system_info():
    """Display system information in sidebar"""
    with st.sidebar:
        st.header("🔧 System Information")
        
        # Memory usage
        memory = psutil.virtual_memory()
        st.metric(
            "Available Memory", 
            f"{memory.available / (1024**3):.1f} GB",
            f"{100 - memory.percent:.1f}% free"
        )
        
        # System status
        if COMPONENTS_AVAILABLE:
            st.success("✅ All components loaded")
        else:
            st.error("❌ Components not available")
        
        # Model information
        st.subheader("📊 Model Information")
        st.info("""
        **Embedding Model**: intfloat/e5-small-v2
        **LLM Model**: google/flan-t5-base
        **Vector Database**: Qdrant Cloud
        **Architecture**: RAG (Retrieval-Augmented Generation)
        """)
        
        # Performance metrics
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        
        st.metric("Total Queries", st.session_state.query_count)

def display_answer_result(result: Dict):
    """Display formatted answer result"""
    answer_result = result['answer_result']
    
    # Main answer
    st.subheader("📝 Answer")
    if answer_result.found_in_document:
        st.success(answer_result.answer)
    else:
        st.warning(answer_result.answer)
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅ Found" if answer_result.found_in_document else "❌ Not Found"
        st.metric("Document Status", status)
    
    with col2:
        st.metric("Confidence", f"{answer_result.confidence_score:.2f}")
    
    with col3:
        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
    
    # Relevant context from document
    if hasattr(answer_result, 'relevant_sentences') and answer_result.relevant_sentences:
        with st.expander("📄 Relevant Context from Document", expanded=True):
            for i, sentence in enumerate(answer_result.relevant_sentences, 1):
                st.markdown(f"**{i}.** {sentence}")
    
    # Citations
    if answer_result.citations:
        with st.expander("📚 Source Citations", expanded=True):
            for i, citation in enumerate(answer_result.citations, 1):
                st.markdown(f"""
                **Citation {i}:**
                - **File**: {citation['filename']}
                - **Page**: {citation['page_number']}
                - **Section**: {citation['section_name']}
                - **Chunk ID**: {citation['chunk_id']}
                """)
    
    # Enhanced citations with full content
    if hasattr(answer_result, 'enhanced_citations') and answer_result.enhanced_citations:
        with st.expander("📋 Detailed Source Content"):
            for i, enhanced_cit in enumerate(answer_result.enhanced_citations, 1):
                with st.container():
                    st.markdown(f"**Source {i}: {enhanced_cit['filename']} - Page {enhanced_cit['page_number']}**")
                    st.markdown(f"*Section: {enhanced_cit['section_name']} | Chunk: {enhanced_cit['chunk_id']} | Words: {enhanced_cit['word_count']}*")
                    
                    # Show content preview or full content
                    if len(enhanced_cit['full_content']) <= 300:
                        st.text_area(f"Full Content {i}:", enhanced_cit['full_content'], height=100, disabled=True)
                    else:
                        st.text_area(f"Content Preview {i}:", enhanced_cit['content_preview'], height=100, disabled=True)
                        
                        if st.button(f"Show Full Content {i}", key=f"show_full_{i}"):
                            st.text_area(f"Complete Content {i}:", enhanced_cit['full_content'], height=200, disabled=True)

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Document-Based RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: 0e1117;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Document-Based RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # System information sidebar
    display_system_info()
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        rag_ui = RAGChatbotUI()
        if not rag_ui.initialize_components():
            st.stop()
    else:
        rag_ui = st.session_state.rag_system
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Your Question")
        
        # Query input
        with st.form("query_form", clear_on_submit=False):
            user_query = st.text_area(
                "Enter your question about the documents:",
                height=100,
                placeholder="e.g., How many players are in a cricket team?"
            )
            
            submitted = st.form_submit_button("🔍 Get Answer", type="primary")
        
        # Process query
        if submitted and user_query.strip():
            st.session_state.query_count += 1
            
            with st.container():
                st.markdown(f'<div class="query-box"><strong>Query:</strong> {user_query}</div>', unsafe_allow_html=True)
                
                # Process the query
                result = rag_ui.process_query(user_query.strip())
                
                if result['success']:
                    display_answer_result(result)
                else:
                    st.error(f"❌ Error: {result['error']}")
                    st.info(f"Processing time: {result['processing_time']:.2f}s")
    
    with col2:
        st.subheader("📖 Usage Instructions")
        st.markdown("""
        **How to use this chatbot:**
        
        1. **Type your question** in the text area
        2. **Click 'Get Answer'** to process
        3. **View the response** with source citations
        4. **Check relevant context** from documents
        5. **Explore detailed sources** in expandable sections
        
        **Example Questions:**
        - How many players are in a cricket team?
        - What are the rules of cricket?
        - What is mentioned about batting?
        
        **Features:**
        ✅ Document-only responses
        ✅ Exact source citations
        ✅ Relevance scoring
        ✅ Context extraction
        ✅ No hallucination
        """)
        
        # Recent queries history
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        if st.session_state.query_history:
            st.subheader("🕒 Recent Queries")
            for i, query in enumerate(reversed(st.session_state.query_history[-5:]), 1):
                st.text(f"{i}. {query}")
    
    # Add current query to history
    if submitted and user_query.strip():
        if user_query not in st.session_state.query_history:
            st.session_state.query_history.append(user_query)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666;">
        <p>🤖 RAG Chatbot System | Built with Streamlit | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Using: FLAN-T5-Base + E5-Small-V2 + Qdrant Vector Database</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    if COMPONENTS_AVAILABLE:
        main()
    else:
        st.error("❌ RAG components not available. Please ensure all dependencies are installed and configured.")
        st.info("Required: document_processor, embedding_engine, vector_db_manager, llm_pipeline modules")
