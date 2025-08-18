import streamlit as st
import os
from langgraph_mcp_bot import app
from module_manager import ModuleManager
from tools.s3_tool import ListBucketsTool
from tools.file_upload import update_faiss_index
from typing import Dict, Any
import tempfile

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'bot' not in st.session_state:
    st.session_state.bot = app  # Use the pre-created app instance
if 'module_manager' not in st.session_state:
    st.session_state.module_manager = ModuleManager()

# App title
st.title("LangGraph MCP Chatbot")

# Module toggles
st.sidebar.header("Enable/Disable Modules")
modules = {
    'rag': 'Knowledge Base (RAG)',
    'sql': 'SQL Database',
    'search': 'Web Search',
    'jira': 'JIRA',
    's3': 'AWS S3'
}

# Create toggles for each module
for module_id, module_name in modules.items():
    enabled = st.sidebar.checkbox(
        module_name,
        value=st.session_state.module_manager.is_enabled(module_id),
        key=f"toggle_{module_id}"
    )
    if enabled:
        st.session_state.module_manager.enable_module(module_id)
    else:
        st.session_state.module_manager.disable_module(module_id)

def handle_file_upload():
    """Handle file upload in the sidebar with chunked processing for large files."""
    uploaded_file = st.sidebar.file_uploader(
        "Upload a file to include in RAG context (up to 20MB)",
        type=["txt", "md", "rst", "docx"]
    )
    
    if uploaded_file:
        try:
            # Get file size
            file_size = len(uploaded_file.getvalue())
            size_mb = file_size / (1024 * 1024)
            
            if size_mb > 20:
                st.sidebar.error(f"File too large: {size_mb:.1f}MB. Maximum size is 20MB")
                return
            
            # Create status containers
            status_container = st.sidebar.empty()
            progress_container = st.sidebar.empty()
            chunk_status = st.sidebar.empty()
            
            try:
                # Phase 1: Process content
                status_container.info("📝 Phase 1/2: Reading file...")
                
                # Extract text based on file type
                if uploaded_file.name.endswith('.docx'):
                    from docx import Document
                    import io
                    doc = Document(io.BytesIO(uploaded_file.getvalue()))
                    content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                else:
                    content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                
                status_container.success(f"✅ Read {uploaded_file.name} ({size_mb:.1f}MB)")
                
                # Phase 2: Process for RAG
                status_container.info("🔄 Phase 2/2: Processing for RAG...")
                
                # Create processing spinner
                with st.spinner("Processing document..."):
                    # Import here to avoid circular imports
                    from tools.rag_tool import RAGTool
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    from langchain_openai import OpenAIEmbeddings
                    from langchain_community.vectorstores import FAISS
                    from langchain.schema import Document
                    
                    # Split into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,  # Increased chunk size
                        chunk_overlap=50,  # Increased overlap
                        length_function=len,
                        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                    )
                    
                    chunks = text_splitter.split_text(content)
                    chunk_status.text(f"Created {len(chunks)} chunks")
                    
                    # Process in larger batches
                    embeddings = OpenAIEmbeddings()
                    batch_size = 20  # Increased batch size
                    db = None
                    
                    # Estimate tokens per chunk
                    avg_tokens_per_chunk = sum(len(chunk.split()) for chunk in chunks[:min(10, len(chunks))]) / min(10, len(chunks))
                    max_tokens = 250000  # OpenAI's limit is 300k, stay under it
                    safe_batch_size = min(batch_size, int(max_tokens / (avg_tokens_per_chunk * 4)))  # 4 tokens per word estimate
                    
                    for i in range(0, len(chunks), safe_batch_size):
                        batch = chunks[i:i + safe_batch_size]
                        docs = [Document(page_content=chunk) for chunk in batch]
                        
                        try:
                            # Create or update index
                            if db is None:
                                db = FAISS.from_documents(docs, embeddings)
                            else:
                                db.add_documents(docs)
                            
                            # Save after each batch
                            db.save_local("faiss_index")
                            
                            # Update progress
                            progress = (i + len(batch)) / len(chunks)
                            progress_container.progress(progress)
                            chunk_status.text(f"Processing chunks: {i + len(batch)}/{len(chunks)}")
                            
                        except Exception as e:
                            st.sidebar.error(f"Error processing batch: {e}")
                            continue
                    
                    if db:
                        st.sidebar.success("✅ Document processed and added to knowledge base")
                        if "rag_chain" in st.session_state:
                            st.session_state["rag_chain"] = None
                    else:
                        st.sidebar.error("❌ Error processing document")
                
            finally:
                # Clear status indicators
                status_container.empty()
                progress_container.empty()
                chunk_status.empty()
            
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
            import traceback
            st.sidebar.error(f"Details:\n{traceback.format_exc()}")            

# File Upload section if RAG is enabled
if st.session_state.module_manager.is_enabled('rag'):
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Upload Files")
    
    handle_file_upload()

# Show S3 bucket selection if S3 is enabled
if st.session_state.module_manager.is_enabled('s3'):
    st.sidebar.markdown("---")
    st.sidebar.header("S3 Buckets")
    
    # List available buckets
    try:
        bucket_tool = ListBucketsTool()
        buckets = bucket_tool._run("")
        if isinstance(buckets, str) and "Error" in buckets:
            st.sidebar.error(f"❌ {buckets}")
        elif not buckets:
            st.sidebar.warning("⚠️ No S3 buckets found")
        else:
            # Add bucket selection
            selected_bucket = st.sidebar.selectbox(
                "Select Bucket",
                buckets,
                key="selected_bucket"
            )
            
            # Add transfer button
            if st.sidebar.button("📥 Transfer to RAG"):
                # Check if RAG is enabled
                if not st.session_state.module_manager.is_enabled('rag'):
                    st.sidebar.warning("⚠️ RAG module must be enabled for transfer")
                    st.sidebar.info("Enabling RAG module...")
                    st.session_state.module_manager.enable_module('rag')
                    st.rerun()
                
                with st.spinner("Transferring documents from S3 to RAG..."):
                    # Call S3 agent directly for transfer
                    from tools.s3_tool import get_s3_agent
                    s3_agent = get_s3_agent()
                    response = s3_agent.invoke({
                        "input": f"transfer documents from {selected_bucket} to rag",
                        "bucket_name": selected_bucket
                    })
                    
                    if isinstance(response, str):
                        final_answer = response
                    else:
                        final_answer = response.get("final_answer", "No response generated")
                    
                    if "Successfully transferred" in final_answer:
                        st.sidebar.success("✅ Transfer complete!")
                        # Extract number of files
                        import re
                        num_files = re.search(r"Successfully transferred (\d+) files", final_answer)
                        if num_files:
                            st.sidebar.info(f"📁 {num_files.group(1)} files transferred")
                    else:
                        st.sidebar.error("❌ Transfer failed")
                        st.sidebar.error(final_answer)
    except Exception as e:
        st.sidebar.error(f"❌ Error listing buckets: {str(e)}")


# Chat interface
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)

# Chat input
if prompt := st.chat_input("Enter your message"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process with bot
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response = st.session_state.bot.invoke({"input": prompt})
            final_answer = response.get("final_answer", "No response generated")
            
            # Check if this was a transfer to RAG operation
            if "transfer" in prompt.lower() and ("rag" in prompt.lower() or "knowledge base" in prompt.lower()):
                if "Successfully transferred" in final_answer:
                    st.success("✅ Documents successfully transferred to knowledge base!")
                    st.info(final_answer)
                else:
                    st.error("❌ Transfer failed. Please check the logs for details.")
                    st.error(final_answer)
            else:
                st.markdown(final_answer)
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
