from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .file_handlers import get_document_loader
import os

def ingest_docs():
    """Process documents from the data directory and update the FAISS index."""
    try:
        # Setup paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        faiss_dir = os.path.join(base_dir, "faiss_index")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize embeddings and text splitter
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Process documents
        docs = []
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            try:
                print(f"\nProcessing file: {file}")
                # Get appropriate loader based on file extension
                if file.endswith(".pdf"):
                    print("Using PDF loader")
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                elif file.endswith(".txt"):
                    print("Using Text loader")
                    loader = TextLoader(file_path)
                    loaded_docs = loader.load()
                else:
                    # Try custom handlers for other file types
                    handler = get_document_loader(file_path)
                    if handler:
                        print(f"Using custom handler for {file}")
                        loaded_docs = handler(file_path)
                        if not loaded_docs:
                            print(f"Warning: No documents returned from handler for {file}")
                    else:
                        print(f"Unsupported file type: {file}")
                        continue
                
                # Split documents into chunks
                if loaded_docs:
                    split_docs = text_splitter.split_documents(loaded_docs)
                    docs.extend(split_docs)
                    print(f"Processed {file} into {len(split_docs)} chunks")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        if not docs:
            print("No documents were processed successfully.")
            return
        
        # Create or update FAISS index
        if os.path.exists(faiss_dir):
            print("Updating existing FAISS index...")
            existing_db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
            existing_db.add_documents(docs)
            existing_db.save_local(faiss_dir)
        else:
            print("Creating new FAISS index...")
            vectorstore = FAISS.from_documents(docs, embeddings)
            os.makedirs(faiss_dir, exist_ok=True)
            vectorstore.save_local(faiss_dir)
        
        print(f"Successfully processed {len(docs)} document chunks.")
        
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        raise

if __name__ == "__main__":
    ingest_docs()
