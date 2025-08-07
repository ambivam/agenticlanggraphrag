import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

class RAGTool:
    def __init__(self, chain):
        self.chain = chain
        self.is_rag_tool = True
    
    def invoke(self, input_text):
        try:
            print(f"\nRAG Tool - Processing query: {input_text}")
            result = self.chain.invoke(input_text)
            print(f"RAG Tool - Got result: {result}")
            return result
        except Exception as e:
            print(f"\nError in RAG Tool invoke: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return None

def get_rag_chain():
    """Get the RAG chain."""
    try:
        print("\n=== Creating RAG Chain ===")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        faiss_dir = os.path.join(base_dir, "faiss_index")
        print(f"FAISS directory: {faiss_dir}")
        
        if not os.path.exists(faiss_dir):
            print(f"Error: FAISS directory not found at {faiss_dir}")
            return None
            
        index_path = os.path.join(faiss_dir, "index.faiss")
        pkl_path = os.path.join(faiss_dir, "index.pkl")
        
        if not (os.path.exists(index_path) and os.path.exists(pkl_path)):
            print(f"Error: Missing required files in {faiss_dir}")
            print(f"index.faiss exists: {os.path.exists(index_path)}")
            print(f"index.pkl exists: {os.path.exists(pkl_path)}")
            return None
        
        print("Loading OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()
        
        print("Loading FAISS index...")
        db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        
        print("Creating retriever...")
        retriever = db.as_retriever()
        
        print("Creating RAG chain...")
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0.7),
            retriever=retriever,
            return_source_documents=True
        )
        
        print("Wrapping chain in RAGTool...")
        return RAGTool(chain)
    except Exception as e:
        print(f"\nError creating RAG chain: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return None
