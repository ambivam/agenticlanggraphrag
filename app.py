import streamlit as st
import os
import tempfile
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
env_sample_path = Path(__file__).parent / '.env.sample'

print(f"Loading .env from: {env_path}")
print(f"Sample env path: {env_sample_path}")

# Try loading .env, fallback to .env.sample
if env_path.exists():
    print("Found .env file, loading...")
    load_dotenv(dotenv_path=env_path)
else:
    print(".env not found, checking .env.sample...")
    if env_sample_path.exists():
        print("Loading from .env.sample")
        load_dotenv(dotenv_path=env_sample_path)
    else:
        print("No environment files found!")

# Import after env vars are loaded
from langgraph_mcp_bot import app
from tools.file_upload import update_faiss_index
from tools.jira_tool import jira_config
from module_manager import module_manager

st.set_page_config(page_title="MCP Chatbot", page_icon="🤖")
st.title("🧠 LangGraph MCP Chatbot")

# Module Settings
st.markdown("### 🔌 Module Settings")

# Create columns for module checkboxes
col1, col2 = st.columns(2)

with col1:
    if st.checkbox("Enable RAG", value=module_manager.is_enabled('rag'), key='rag_module'):
        module_manager.enable_module('rag')
    else:
        module_manager.disable_module('rag')
        
    if st.checkbox("Enable SQL", value=module_manager.is_enabled('sql'), key='sql_module'):
        module_manager.enable_module('sql')
    else:
        module_manager.disable_module('sql')

with col2:
    if st.checkbox("Enable Search", value=module_manager.is_enabled('search'), key='search_module'):
        module_manager.enable_module('search')
    else:
        module_manager.disable_module('search')
        
    if st.checkbox("Enable JIRA", value=module_manager.is_enabled('jira'), key='jira_module'):
        module_manager.enable_module('jira')
    else:
        module_manager.disable_module('jira')

# Show enabled modules status
enabled_modules = [mod for mod, enabled in module_manager.modules.items() if enabled]
if enabled_modules:
    st.success(f"Enabled modules: {' ➜ '.join(enabled_modules).upper()}")
else:
    st.warning("No modules enabled. Please enable at least one module above.")

st.markdown("---")

# RAG Section
if module_manager.is_enabled('rag'):
    st.markdown("### 📚 RAG Module")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Knowledge Base Files",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "xlsx", "xls", "csv", "pptx", "md", "json"],
        help="Upload knowledge base documents"
    )
    
    # Show supported formats
    with st.expander("Supported Formats"):
        st.markdown("""
            - Documents: PDF (.pdf), Text (.txt), Word (.docx)
            - Data Files: Excel (.xlsx, .xls), CSV (.csv)
            - Presentations: PowerPoint (.pptx)
            - Other: Markdown (.md), JSON (.json)
            
            **Size Limit:** 200MB per file
        """)
    
    # RAG Query
    rag_query = st.text_input("RAG Query", placeholder="Ask a question about your documents...")
    if st.button("Search Knowledge Base") and rag_query:
        with st.spinner("Searching knowledge base..."):
            response = app.invoke({"input": rag_query})
            if response.get("final_answer"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join("output", f"rag_response_{timestamp}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"Question: {rag_query}\n\nAnswer: {response['final_answer']}")
                st.info(f"💾 Response saved to: {output_file}")
else:
    uploaded_files = None

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save uploaded file to data directory
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    if st.button("Process Files"):
        with st.spinner("Processing files..."):
            # Save uploaded files temporarily
            temp_files = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                    tmp_file.write(file.getvalue())
                    temp_files.append(tmp_file.name)
            
            # Update FAISS index
            num_chunks, status = update_faiss_index(temp_files)
            
            # Clean up temporary files
            for temp_file in temp_files:
                os.unlink(temp_file)
            
            if num_chunks > 0:
                st.success(f"✅ {status} Added {num_chunks} chunks to the knowledge base.")
            else:
                st.error(f"❌ {status}")

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Other Module Sections

# SQL Module
if module_manager.is_enabled('sql'):
    st.markdown("### 🔎 SQL Module")
    sql_query = st.text_input("SQL Query", placeholder="Enter your database query...")
    if st.button("Run SQL Query") and sql_query:
        with st.spinner("Running SQL query..."):
            response = app.invoke({"input": sql_query})
            if response.get("sql_context"):
                st.code(response["sql_context"], language="sql")

# Search Module
if module_manager.is_enabled('search'):
    st.markdown("### 🔍 Web Search Module")
    search_query = st.text_input("Search Query", placeholder="Enter your search query...")
    if st.button("Search Web") and search_query:
        with st.spinner("Searching the web..."):
            response = app.invoke({"input": search_query})
            if response.get("serp_context"):
                st.write(response["serp_context"])

# JIRA Module
if module_manager.is_enabled('jira'):
    st.markdown("### 🎫 JIRA Module")
    
    # Initialize config in session state
    if 'jira_config' not in st.session_state:
        # Try to load from environment first
        st.session_state.jira_config = {
            'url': os.getenv("JIRA_URL", ""),
            'username': os.getenv("JIRA_USERNAME", ""),
            'token': os.getenv("JIRA_API_TOKEN", ""),
            'project_key': os.getenv("JIRA_PROJECT_KEY", ""),
            'openai_key': os.getenv("OPENAI_API_KEY", "")
        }
        print("\nInitial JIRA config from env:")
        print(f"URL: {st.session_state.jira_config['url']}")
        print(f"Username: {st.session_state.jira_config['username']}")
        print(f"Project Key: {st.session_state.jira_config['project_key']}")
    
    # JIRA Configuration
    with st.expander("JIRA Configuration"):
        # Show current config status
        if jira_config.is_configured():
            st.success("✅ JIRA is configured")
        else:
            st.warning("⚠️ JIRA needs configuration")
        
        # Configuration form
        with st.form("jira_config_form"):
            new_url = st.text_input(
                "JIRA URL", 
                value=st.session_state.jira_config['url'],
                placeholder="https://your-domain.atlassian.net"
            )
            new_username = st.text_input(
                "JIRA Username", 
                value=st.session_state.jira_config['username'],
                placeholder="your.email@company.com"
            )
            new_token = st.text_input(
                "JIRA API Token", 
                value=st.session_state.jira_config['token'],
                type="password"
            )
            new_project_key = st.text_input(
                "Default Project Key", 
                value=st.session_state.jira_config['project_key'],
                placeholder="e.g., PROJ"
            )
            
            st.markdown("### OpenAI Configuration")
            st.info("OpenAI API key is required for test case generation")
            
            new_openai_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.jira_config['openai_key'],
                type="password",
                help="Get your API key from https://platform.openai.com/account/api-keys"
            )
            
            if st.form_submit_button("Save Configuration"):
                print("\nSaving JIRA Configuration:")
                
                # Update session state
                st.session_state.jira_config.update({
                    'url': new_url,
                    'username': new_username,
                    'token': new_token,
                    'project_key': new_project_key,
                    'openai_key': new_openai_key
                })
                
                print(f"URL: {new_url}")
                print(f"Username: {new_username}")
                print(f"Project Key: {new_project_key}")
                
                # Create .env file if it doesn't exist
                env_path = Path(__file__).parent / '.env'
                
                # Get existing OpenAI key if present
                openai_key = os.getenv('OPENAI_API_KEY', '')
                
                env_content = f"""# JIRA Configuration
JIRA_URL={new_url}
JIRA_USERNAME={new_username}
JIRA_API_TOKEN={new_token}
JIRA_PROJECT_KEY={new_project_key}

# OpenAI Configuration
OPENAI_API_KEY={openai_key}
"""
                
                print(f"Writing config to: {env_path}")
                with open(env_path, 'w') as f:
                    f.write(env_content)
                
                # Force reload environment
                load_dotenv(dotenv_path=env_path, override=True)
                
                # Update JIRA config object
                jira_config.reload_config()
                
                if jira_config.is_configured():
                    st.success("✅ JIRA configuration saved!")
                else:
                    st.error("❌ Please fill in all JIRA configuration fields")
    
    # JIRA Search
    col1, col2 = st.columns([2, 1])
    with col1:
        jira_query = st.text_input("JIRA Query", placeholder="Search JIRA issues...")
    with col2:
        search_project = st.text_input("Override Project Key", placeholder="e.g., PROJ")
    
    if st.button("Search JIRA") and jira_query:
        if not jira_config.is_configured():
            st.error("❌ Please configure JIRA settings first")
        else:
            if search_project:
                jira_config.set_project_key(search_project)
            with st.spinner("Searching JIRA..."):
                response = app.invoke({"input": jira_query})
                if response.get("jira_context"):
                    st.write(response["jira_context"])
                    if response.get("test_cases"):
                        st.markdown("### 🧪 Generated Test Cases")
                        st.markdown(response["test_cases"])

# Set default values
llm_temperature = 0.7
scenarios_per_category = 10
