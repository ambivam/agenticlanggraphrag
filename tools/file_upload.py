import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import sys
from typing import List, Tuple

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.file_handlers import get_document_loader

def load_pdf(file_path: str) -> List[str]:
    """Load a PDF file using PyPDF2."""
    try:
        import pypdf
    except ImportError:
        raise ImportError(
            "pypdf package not found, please install it with 'pip install pypdf'"
        )
    
    # Create a PDF reader object
    with open(file_path, 'rb') as file:
        pdf = pypdf.PdfReader(file)
        text = ''
        # Extract text from each page
        for page in pdf.pages:
            text += page.extract_text() + '\n'
        
        # Create a document with metadata
        from langchain_core.documents import Document
        return [Document(
            page_content=text,
            metadata={"source": file_path}
        )]

def process_file(file_path: str) -> List[str]:
    """Process a file and return its chunks."""
    try:
        # Determine loader based on file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Load document based on type
        if ext == '.pdf':
            documents = load_pdf(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
            documents = loader.load()
        else:
            # Try custom handlers for other file types
            handler = get_document_loader(file_path)
            if handler:
                documents = handler(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        raise

def update_faiss_index(file_paths: List[str]) -> Tuple[int, str]:
    """Update FAISS index with new documents."""
    try:
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
        for file_path in file_paths:
            try:
                # Load document based on type
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                elif file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                    loaded_docs = loader.load()
                else:
                    # Try custom handlers for other file types
                    handler = get_document_loader(file_path)
                    if handler:
                        loaded_docs = handler(file_path)
                    else:
                        print(f"Unsupported file type: {file_path}")
                        continue
                
                # Split documents into chunks
                if loaded_docs:
                    split_docs = text_splitter.split_documents(loaded_docs)
                    docs.extend(split_docs)
                    print(f"Processed {os.path.basename(file_path)} into {len(split_docs)} chunks")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        if not docs:
            return 0, "No valid documents were processed."
        
        # Update or create FAISS index
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        faiss_dir = os.path.join(base_dir, "faiss_index")
        
        if os.path.exists(faiss_dir):
            db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(docs)
            db.save_local(faiss_dir)
            return len(docs), "Successfully updated knowledge base."
        else:
            os.makedirs(faiss_dir, exist_ok=True)
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(faiss_dir)
            return len(docs), "Successfully created knowledge base."
            
    except Exception as e:
        print(f"Error updating FAISS index: {str(e)}")
        return 0, f"Error: {str(e)}"
