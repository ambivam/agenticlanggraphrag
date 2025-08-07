import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
env_path = Path(__file__).parent / '.env'
env_sample_path = Path(__file__).parent / '.env.sample'

print(f"Loading .env from: {env_path}")
print(f"Sample env path: {env_sample_path}")

# Try loading .env, fallback to .env.sample
if env_path.exists():
    print("Found .env file, loading...")
    load_dotenv(dotenv_path=env_path, override=True)
    # Print AWS credentials for debugging
    print(f"AWS_ACCESS_KEY_ID: {'✓ Set' if os.getenv('AWS_ACCESS_KEY_ID') else '✗ Not set'}")
    print(f"AWS_SECRET_ACCESS_KEY: {'✓ Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else '✗ Not set'}")
    print(f"AWS_REGION: {os.getenv('AWS_REGION', 'Not set')}")
else:
    print(".env not found, checking .env.sample...")
    if env_sample_path.exists():
        print("Loading from .env.sample")
        load_dotenv(dotenv_path=env_sample_path, override=True)
    else:
        print("No environment files found!")

# Now import other modules after env vars are loaded
import streamlit as st
import tempfile
from datetime import datetime
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
        
    if st.checkbox("Enable S3", value=module_manager.is_enabled('s3'), key='s3_module'):
        module_manager.enable_module('s3')
    else:
        module_manager.disable_module('s3')

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
            # Save response to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save both question and response in a single file
            output_file = os.path.join(output_dir, f"rag_query_{timestamp}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Question: {rag_query}\n\nResponse:\n{response.get('final_answer', '')}")
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

# S3 Module
if module_manager.is_enabled('s3'):
    print("\n=== S3 Module Enabled ===")
    st.markdown("### 📦 S3 Module")
    
    # Initialize S3 client
    try:
        print("Initializing S3 client...")
        from tools.s3_tool import ListBucketsTool
        s3_tool = ListBucketsTool()
        
        print("\nListing buckets...")
        # Get list of buckets
        buckets = s3_tool._run("")
        print(f"Buckets response: {buckets}")
        
        if isinstance(buckets, str) and 'Error' in buckets:
            print(f"Error listing buckets: {buckets}")
            st.error(f"❌ {buckets}")
        else:
            # Bucket selector
            selected_bucket = st.selectbox(
                "Select S3 Bucket",
                options=buckets,
                help="Choose the S3 bucket to interact with"
            )
            
            if selected_bucket:
                # Query input
                s3_query = st.text_input(
                    "S3 Query",
                    placeholder="Ask about your S3 files (e.g., 'list all PDF files', 'read contents of config.json')")
                
                if st.button("Query S3") and s3_query:
                    print(f"\n=== Processing S3 Query ===")
                    with st.spinner("Processing S3 query..."):
                        # Format query for S3 module
                        formatted_query = s3_query.strip()
                        
                        # Create state for langgraph
                        state = {
                            "input": formatted_query,
                            "bucket": selected_bucket,
                            "found": False,
                            "rag_context": None,
                            "sql_context": None,
                            "serp_context": None,
                            "jira_context": None,
                            "s3_context": None,
                            "test_cases": None,
                            "final_answer": None
                        }
                        
                        print(f"Query details:")
                        print(f"- Raw query: {s3_query}")
                        print(f"- Formatted query: {formatted_query}")
                        print(f"- Selected bucket: {selected_bucket}")
                        print(f"- Full state: {state}")
                        
                        # Invoke app with state
                        print("\nInvoking langgraph...")
                        response = app.invoke(state)
                        print(f"S3 agent response: {response}")
                        
                        if response.get("final_answer"):
                            # Display response
                            st.markdown("### Results")
                            st.write(response["final_answer"])
                            
                            # Save response to a single file
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_file = os.path.join("output", f"response_{timestamp}.txt")
                            with open(output_file, "w", encoding="utf-8") as f:
                                f.write(f"Module: S3\nBucket: {selected_bucket}\nQuery: {s3_query}\n\nResponse:\n{response['final_answer']}")
                            st.info(f"💾 Response saved to: {output_file}")
    except Exception as e:
        st.error(f"❌ Error initializing S3 module: {str(e)}. Please check your AWS credentials.")

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
