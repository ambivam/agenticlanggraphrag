import streamlit as st
import os
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
    col1, col2 = st.columns([2, 1])
    with col1:
        jira_query = st.text_input("JIRA Query", placeholder="Search JIRA issues...")
    with col2:
        jira_project = st.text_input("Project Key", placeholder="e.g., PROJ")
    
    if st.button("Search JIRA") and jira_query:
        if jira_project:
            jira_config.set_project_key(jira_project)
        with st.spinner("Searching JIRA..."):
            response = app.invoke({"input": jira_query})
            if response.get("jira_context"):
                st.write(response["jira_context"])

# Set default values
llm_temperature = 0.7
scenarios_per_category = 10
