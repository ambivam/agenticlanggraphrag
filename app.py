import streamlit as st
import os
import tempfile
from langgraph_mcp_bot import app
from tools.file_upload import update_faiss_index
from tools.jira_tool import jira_config

st.set_page_config(page_title="MCP Chatbot", page_icon="🤖")
st.title("🧠 LangGraph MCP Chatbot")

# File upload section
st.markdown("### 📚 Upload Knowledge Base")
st.caption("Upload PDF or text files to add to the chatbot's knowledge base")

uploaded_files = st.file_uploader(
    "Choose files",
    accept_multiple_files=True,
    type=["pdf", "txt", "docx", "xlsx", "xls", "csv", "pptx", "md", "json"],
    help="Upload knowledge base documents (PDF, TXT, Word, Excel, CSV, PowerPoint, Markdown, or JSON)"
)

# Show supported formats and size limit
st.markdown("""
    #### Supported Formats:
    - Documents: PDF (.pdf), Text (.txt), Word (.docx)
    - Data Files: Excel (.xlsx, .xls), CSV (.csv)
    - Presentations: PowerPoint (.pptx)
    - Other: Markdown (.md), JSON (.json)
    
    **Size Limit:** 200MB per file
""")

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

# Divider
st.markdown("---")

# JIRA Project Key Input
st.markdown("### 🎫 JIRA Query")
with st.expander("JIRA Search Settings", expanded=True):
    jira_project = st.text_input("JIRA Project Key", placeholder="e.g., PROJ")
    st.caption("Enter the project key to search for issues in that project")
    st.caption("You can also enter a natural language query to search across issues")

st.markdown("---")

# Chat section
st.markdown("### 💬 Chat")
st.caption("Order of sources: RAG ➜ MySQL ➜ Web Search (SerpAPI) ➜ JIRA")

user_input = st.text_input("Enter your question:")

if user_input:
    with st.spinner("Searching..."):
        # Set JIRA project key if provided
        if jira_project:
            jira_config.set_project_key(jira_project)
        
        state = {"input": user_input}
        result = app.invoke(state)
        st.markdown("### 🤖 Response")
        
        # Display results from each source in expandable sections
        if result.get("rag_context"):
            with st.expander("📚 Knowledge Base Results", expanded=True):
                st.markdown(result["rag_context"])
        
        if result.get("sql_context"):
            with st.expander("💾 Database Results", expanded=True):
                st.markdown(result["sql_context"])
        
        if result.get("serp_context"):
            with st.expander("🔍 Web Search Results", expanded=True):
                st.markdown(result["serp_context"])
                
        if result.get("jira_context"):
            with st.expander("🎫 JIRA Results", expanded=True):
                st.markdown(result["jira_context"])
                
        if result.get("test_cases"):
            with st.expander("🧪 Test Cases", expanded=True):
                st.markdown(result["test_cases"], unsafe_allow_html=True)
