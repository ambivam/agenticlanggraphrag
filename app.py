import streamlit as st
import os
import tempfile
from langgraph_mcp_bot import app
from tools.file_upload import update_faiss_index

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

# Chat section
st.markdown("### 💬 Chat")
st.caption("Order of sources: RAG ➜ MySQL ➜ Web Search (SerpAPI)")

user_input = st.text_input("Enter your question:")

if user_input:
    with st.spinner("Searching..."):
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
