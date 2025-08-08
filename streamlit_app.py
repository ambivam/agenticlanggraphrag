import streamlit as st
import os
from langgraph_mcp_bot import app
from module_manager import ModuleManager
from tools.s3_tool import ListBucketsTool
from typing import Dict, Any

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
