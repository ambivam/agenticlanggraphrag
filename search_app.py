import streamlit as st
import os
import json
from datetime import datetime
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from config import Config
from dotenv import load_dotenv
from main import SearchOrchestrator

# Load environment variables
load_dotenv(override=True)

# Page config
st.set_page_config(page_title="Multi-Source Search", page_icon="🔍")
st.title("🔍 Multi-Source Search System")

# Initialize session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = SearchOrchestrator()

# Always ensure RAG module has embeddings initialized
if Config.OPENAI_API_KEY:
    st.session_state.orchestrator.rag.embeddings = OpenAIEmbeddings(
        openai_api_key=Config.OPENAI_API_KEY,
        model="text-embedding-ada-002"
    )
    
    # Try to load existing vector store
    vector_store_path = st.session_state.orchestrator.rag.get_vector_store_path()
    index_path = os.path.join(vector_store_path, "index.faiss")
    
    if os.path.exists(index_path):
        try:
            st.session_state.orchestrator.rag.vector_store = FAISS.load_local(
                vector_store_path,
                st.session_state.orchestrator.rag.embeddings
            )
            st.sidebar.success("✅ Vector store loaded successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load vector store: {str(e)}")
    else:
        st.sidebar.info("ℹ️ No existing vector store found. Please upload documents.")
else:
    st.sidebar.error("❌ OpenAI API key not found. Please set it in your environment variables.")
    
# Debug information about environment
st.sidebar.markdown("### Environment Status")
debug_info = Config.debug_info()

# Show environment status
st.sidebar.markdown("#### Environment File")
if debug_info["env_file_exists"]:
    st.sidebar.success(f"✅ .env file found at:\n{debug_info['env_file_path']}")
else:
    st.sidebar.error(f"❌ .env file not found at:\n{debug_info['env_file_path']}")

# Show OpenAI key status
st.sidebar.markdown("#### OpenAI API Key")
if debug_info["openai_key_exists"]:
    st.sidebar.success("✅ OpenAI API Key found in environment")
else:
    st.sidebar.error("❌ OpenAI API Key not found in environment")

# Show loaded environment variables
st.sidebar.markdown("#### Loaded Environment Variables")
st.sidebar.code("\n".join(debug_info["loaded_vars"]))

# Module Selection
st.markdown("### 📋 Enable/Disable Modules")

# Create columns for module checkboxes
col1, col2 = st.columns(2)

with col1:
    if st.checkbox("RAG Search", key="rag_enabled"):
        st.session_state.orchestrator.module_manager.enable_module("rag")
    else:
        st.session_state.orchestrator.module_manager.disable_module("rag")
        
    if st.checkbox("Web Search (SerpAPI)", key="web_search_enabled"):
        st.session_state.orchestrator.module_manager.enable_module("web_search")
    else:
        st.session_state.orchestrator.module_manager.disable_module("web_search")

with col2:
    if st.checkbox("MySQL Search", key="mysql_enabled"):
        st.session_state.orchestrator.module_manager.enable_module("mysql")
    else:
        st.session_state.orchestrator.module_manager.disable_module("mysql")
        
    if st.checkbox("JIRA Search", key="jira_enabled"):
        st.session_state.orchestrator.module_manager.enable_module("jira")
    else:
        st.session_state.orchestrator.module_manager.disable_module("jira")

# Configuration Section
st.markdown("### ⚙️ Module Configuration")

with st.expander("Configuration Settings", expanded=False):
    # RAG Configuration
    if st.session_state.orchestrator.module_manager.is_module_enabled("rag"):
        st.markdown("#### RAG Configuration")
        openai_api_key = st.text_input("OpenAI API Key", value=Config.OPENAI_API_KEY, type="password", key="rag_openai_api_key")
        st.caption("Required for document embeddings and similarity search. Set in .env file or override here.")
        
        # File upload section
        st.markdown("##### 📚 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files to add to knowledge base",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "doc", "csv"],
            help="Upload documents to be processed and added to the FAISS vector store"
        )
        
        # Show supported formats
        st.markdown("""
        **Supported Formats:**
        - Documents: PDF (.pdf), Text (.txt), Word (.docx, .doc)
        - Data: CSV (.csv)
        """)
        
        if uploaded_files:
            process_button = st.button("Process Files", key="process_files_button")
            if process_button:
                # Get OpenAI key from environment or UI
                openai_key = Config.OPENAI_API_KEY or st.session_state.get("openai_api_key")
                
                # Debug information
                if not openai_key:
                    st.error("❌ OpenAI API key not found in .env file or UI input")
                    st.info("Debug: Environment variables loaded = " + ", ".join([k for k in os.environ.keys() if not k.startswith("_")]))
                else:
                    with st.spinner("Processing files..."):
                        # Initialize RAG if not already initialized
                        if not st.session_state.orchestrator.rag.embeddings:
                            with st.spinner("Initializing OpenAI embeddings..."):
                                init_success = st.session_state.orchestrator.rag.initialize(
                                    documents_path="",  # Not using directory path
                                    openai_api_key=openai_key
                                )
                                # Show debug info
                                if st.session_state.orchestrator.rag._debug_info:
                                    st.expander("Debug Info", expanded=True).code(
                                        "\n".join(st.session_state.orchestrator.rag._debug_info)
                                    )
                                    
                                if not init_success:
                                    st.error("❌ Failed to initialize OpenAI embeddings. Please verify your API key is valid.")
                                    st.info("Debug: Make sure your .env file contains OPENAI_API_KEY=sk-...")
                                    st.stop()
                                    
                        # Process the uploaded files
                        files_content = [file.read() for file in uploaded_files]
                        filenames = [file.name for file in uploaded_files]
                        
                        num_chunks, status = st.session_state.orchestrator.rag.process_uploaded_files(
                            files_content, filenames
                        )
                        
                        if num_chunks > 0:
                            st.success(f"✅ {status}")
                        else:
                            st.error(f"❌ {status}")
                        

        
    # MySQL Configuration
    if st.session_state.orchestrator.module_manager.is_module_enabled("mysql"):
        st.markdown("#### MySQL Configuration")
        mysql_host = st.text_input("Host", value=Config.MYSQL_HOST, key="mysql_host_input")
        mysql_user = st.text_input("User", value=Config.MYSQL_USER, key="mysql_user_input")
        mysql_password = st.text_input("Password", value=Config.MYSQL_PASSWORD, type="password", key="mysql_password_input")
        mysql_database = st.text_input("Database", value=Config.MYSQL_DATABASE, key="mysql_database_input")
        st.caption("Configure in .env file or override here.")
        
    # JIRA Configuration
    if st.session_state.orchestrator.module_manager.is_module_enabled("jira"):
        st.markdown("#### JIRA Configuration")
        jira_server = st.text_input("JIRA Server", value=Config.JIRA_SERVER, key="jira_server_input")
        jira_email = st.text_input("Email", value=Config.JIRA_EMAIL, key="jira_email_input")
        jira_token = st.text_input("API Token", value=Config.JIRA_API_TOKEN, type="password", key="jira_token_input")
        st.caption("Configure in .env file or override here.")

# Search Section
st.markdown("### 🔍 Search")

# Check if any search module is enabled
if not any([
    st.session_state.orchestrator.module_manager.is_module_enabled(module)
    for module in ["rag", "mysql", "web_search", "jira"]
]):
    st.error("Please enable at least one search module")
else:
    query = st.text_input("Enter your search query", key="search_query_input")
    search_button = st.button("Search", key="search_button")
    
    if search_button and query:
        st.markdown("### 📝 Search Results")
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        
        # RAG Results
        if st.session_state.orchestrator.module_manager.is_module_enabled("rag"):
            with st.expander("RAG Results", expanded=True):
                st.text(f"[{timestamp}]")
                
                if not st.session_state.orchestrator.rag.embeddings:
                    st.warning("⚠️ Please upload and process some documents first")
                elif not st.session_state.orchestrator.rag.vector_store:
                    st.warning("⚠️ Vector store not initialized. Please upload and process some documents.")
                else:
                    with st.spinner("Searching documents..."):
                        result = st.session_state.orchestrator.rag.query(query)
                        
                        # Show debug info
                        if st.session_state.orchestrator.rag._debug_info:
                            st.expander("Debug Info", expanded=False).code(
                                "\n".join(st.session_state.orchestrator.rag._debug_info)
                            )
                        
                        if result:
                            if "Error:" in result:
                                st.error(result)
                            else:
                                st.markdown(result)
                        else:
                            st.info("No results found in the documents.")
    
        # MySQL Results
        if st.session_state.orchestrator.module_manager.is_module_enabled("mysql"):
            with st.expander("MySQL Results", expanded=True):
                st.text(f"[{timestamp}]")
                with st.spinner("Searching MySQL..."):
                    result = st.session_state.orchestrator.mysql.query(query)
                    if result:
                        st.code(result, language="sql")
                    else:
                        st.info("No results found in MySQL database.")
        
        # Web Search Results
        if st.session_state.orchestrator.module_manager.is_module_enabled("web_search"):
            with st.expander("Web Search Results", expanded=True):
                st.text(f"[{timestamp}]")
                with st.spinner("Searching the web..."):
                    result = st.session_state.orchestrator.web_search.search(query)
                    if result:
                        st.markdown(result)
                    else:
                        st.info("No web search results found.")
        
        # JIRA Results
        if st.session_state.orchestrator.module_manager.is_module_enabled("jira"):
            with st.expander("JIRA Results", expanded=True):
                st.text(f"[{timestamp}]")
                with st.spinner("Searching JIRA..."):
                    result = st.session_state.orchestrator.jira.search(query)
                    if result:
                        st.markdown(result)
                    else:
                        st.info("No JIRA issues found.")

# Cleanup on session end
def cleanup():
    if hasattr(st.session_state, 'orchestrator'):
        st.session_state.orchestrator.cleanup()

# Register cleanup
st.session_state['cleanup_registered'] = True
