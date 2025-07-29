import os
from typing import List, Tuple
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

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
    """Update FAISS index with new documents.
    Returns: (number of chunks added, status message)
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        faiss_dir = os.path.join(base_dir, "faiss_index")
        
        # Process all files
        all_chunks = []
        for file_path in file_paths:
            chunks = process_file(file_path)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return 0, "No content found in the uploaded files."
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Load existing index if it exists, otherwise create new
        if os.path.exists(faiss_dir):
            db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(all_chunks)
        else:
            db = FAISS.from_documents(all_chunks, embeddings)
            os.makedirs(faiss_dir, exist_ok=True)
        
        # Save updated index
        db.save_local(faiss_dir)
        
        return len(all_chunks), "Successfully added to the knowledge base."
    except Exception as e:
        return 0, f"Error updating index: {str(e)}"
