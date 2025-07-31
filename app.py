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

# JIRA and LLM Settings
st.markdown("### ⚙️ Settings")
with st.expander("Search and Generation Settings", expanded=True):
    # JIRA Project Key
    st.markdown("#### 🎫 JIRA Project")
    jira_project = st.text_input("Project Key", placeholder="e.g., PROJ")
    st.caption("Enter the project key to search for issues in that project")
    
    # Add controls column
    col1, col2 = st.columns(2)
    
    # Add temperature slider
    with col1:
        llm_temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower values (0.0) give more focused and consistent outputs. Higher values (1.0) give more creative and varied outputs."
        )
    
    # Add test case count slider
    with col2:
        scenarios_per_category = st.slider(
            "Scenarios per Category",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            help="Number of test scenarios to generate per category. More scenarios = more comprehensive testing but longer generation time."
        )

st.markdown("---")

# Chat section
st.markdown("### 💬 Chat")
st.caption("Order of sources: RAG ➜ MySQL ➜ Web Search (SerpAPI) ➜ JIRA")

# Search input and button in columns for better layout
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Enter your question:", key="search_input")
with col2:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Only execute search when button is clicked and input is not empty
if search_button and user_input:
    with st.spinner("Searching..."):
        # Set JIRA project key if provided
        if jira_project:
            jira_config.set_project_key(jira_project)
            
        # Update state with project key, temperature, and test cases count
        state = {
            "input": user_input,
            "llm_temperature": llm_temperature,
            "scenarios_per_category": scenarios_per_category
        }
        
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

elif search_button and not user_input:
    st.error("⚠️ Please enter a question before searching")
