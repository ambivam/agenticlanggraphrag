import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

class RAGTool:
    def __init__(self, chain):
        self.chain = chain
        self.is_rag_tool = True
    
    def invoke(self, input_text):
        return self.chain.invoke(input_text)

def get_rag_chain():
    """Get the RAG chain."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        faiss_dir = os.path.join(base_dir, "faiss_index")
        
        embeddings = OpenAIEmbeddings()
        db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()
        
        # Create RAG chain
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0.7),
            retriever=retriever,
            return_source_documents=True
        )
        
        # Wrap chain in our custom class
        return RAGTool(chain)
    except Exception as e:
        print(f"Error creating RAG chain: {str(e)}")
        return None
