import streamlit as st
import os
import tempfile
from langgraph_mcp_bot import app
from tools.file_upload import update_faiss_index
from tools.jira_tool import jira_config

st.set_page_config(page_title="MCP Chatbot", page_icon="🤖")
st.title("🧠 LangGraph MCP Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input
if user_question := st.chat_input("Ask your question here..."):
    # Set JIRA project key if provided
    if jira_project:
        jira_config.set_project_key(jira_project)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_question)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Check if this is a test case generation request
            is_test_case_request = any(x in user_question.lower() for x in [
                "generate test", "bdd", "gherkin", "scenario", "test case"
            ])
            
            # Check if this is a continuation request
            continue_previous = any(x in user_question.lower() for x in [
                "more", "additional", "continue", "generate more"
            ])
            
            # Invoke the LangGraph app with appropriate state
            response = app.invoke({
                "input": user_question,
                "query": user_question,  # Used by test_case_node
                "continue_previous": continue_previous if is_test_case_request else False,
                "llm_temperature": llm_temperature,
                "scenarios_per_category": scenarios_per_category
            })
            
            # Extract the final answer
            answer = response.get("final_answer", "I'm sorry, I couldn't process your request.")
            
            # Display the response
            st.markdown(answer, unsafe_allow_html=True)
            
            # Add assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
