from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PDFMinerLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
import os
from typing import List, Dict, Any
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
    
    # Load documents from multiple file types
    print("Loading documents...")
    loaders = [
        ("**/*.txt", TextLoader),
        ("**/*.pdf", PDFMinerLoader),
        ("**/*.docx", Docx2txtLoader),
        ("**/*.md", UnstructuredMarkdownLoader)
    ]
    
    documents = []
    for glob_pattern, loader_cls in loaders:
        try:
            loader = DirectoryLoader(data_dir, glob=glob_pattern, loader_cls=loader_cls)
            docs = loader.load()
            print(f"Loaded {len(docs)} documents from {glob_pattern}")
            if docs:
                print("Sample files:")
                for doc in docs[:3]:  # Show first 3 files
                    print(f"- {doc.metadata.get('source', 'Unknown source')}")
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {glob_pattern}: {str(e)}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    
    if not documents:
        raise ValueError("No documents found in the data directory!")
    
    # Split documents with better chunking
    print("\nSplitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks for better retrieval
        chunk_overlap=100,  # Decent overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # More granular splitting
        keep_separator=True
    )
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks")
    
    # Debug: Show sample chunks
    print("\nSample chunks:")
    for i, chunk in enumerate(texts[:3], 1):  # Show first 3 chunks
        print(f"\nChunk {i}:")
        print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
        print(f"Content preview: {chunk.page_content[:200]}...")
    
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
