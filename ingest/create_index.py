from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os
from dotenv import load_dotenv

load_dotenv()

def create_faiss_index():
    print("Starting FAISS index creation...")
    
    # Get absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    faiss_dir = os.path.join(base_dir, "faiss_index")
    
    print(f"Base directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"FAISS directory: {faiss_dir}")
    
    # Create data directory and sample document
    if not os.path.exists(data_dir):
        print(f"Creating data directory at {data_dir}")
        os.makedirs(data_dir)
    
    # Always create/update the sample document
    sample_file = os.path.join(data_dir, "sample.txt")
    print(f"Creating sample document at {sample_file}")
    with open(sample_file, "w", encoding='utf-8') as f:
        f.write("""This is a sample document for the MCP chatbot.
        The chatbot uses multiple sources to answer questions:
        1. RAG (Retrieval Augmented Generation) with FAISS
        2. MySQL database queries
        3. Web search using SerpAPI
        
        The system tries each source in sequence until it finds a good answer.
        """)
    
    # Load documents
    print("Loading documents...")
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    if not documents:
        raise ValueError("No documents found in the data directory!")
    
    # Split documents
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks")
    
    # Create and save FAISS index
    print("Creating embeddings and FAISS index...")
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    
    # Create faiss_index directory if it doesn't exist
    if not os.path.exists(faiss_dir):
        print(f"Creating FAISS index directory at {faiss_dir}")
        os.makedirs(faiss_dir)
    
    # Save the index
    print("Saving FAISS index...")
    db.save_local(faiss_dir)
    print("FAISS index created successfully!")

if __name__ == "__main__":
    create_faiss_index()
