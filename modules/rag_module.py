import os
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
import tempfile

class RAGModule:
    SUPPORTED_EXTENSIONS = {
        '.txt': TextLoader,
        '.pdf': PyPDFLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.csv': CSVLoader
    }
    
    # Vector store persistence path
    VECTOR_STORE_PATH = "faiss_index"
    
    @staticmethod
    def get_vector_store_path():
        """Get the absolute path to the vector store directory"""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "faiss_index"))
    
    def __init__(self):
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        self._debug_info = []  # Store debug info for UI feedback
        
    def _get_loader_for_file(self, file_path: str) -> Optional[BaseLoader]:
        """Get the appropriate loader for a file based on its extension"""
        ext = Path(file_path).suffix.lower()
        self._debug_info.append(f"File extension: {ext}")
        loader_class = self.SUPPORTED_EXTENSIONS.get(ext)
        if loader_class:
            self._debug_info.append(f"Found loader: {loader_class.__name__}")
        else:
            self._debug_info.append(f"No loader found for extension: {ext}")
        return loader_class
        
    def process_uploaded_files(self, files: List[bytes], filenames: List[str]) -> Tuple[int, str]:
        """Process uploaded files and add them to the vector store"""
        self._debug_info = []  # Reset debug info
        vector_store_path = self.get_vector_store_path()
        
        try:
            if not self.embeddings:
                self._debug_info.append("OpenAI embeddings not initialized")
                return 0, "Error: OpenAI embeddings not initialized"
            
            self._debug_info.append(f"Processing {len(files)} files: {', '.join(filenames)}")
            
            documents = []
            temp_files = []
            
            # Save files temporarily and load them
            for file_content, filename in zip(files, filenames):
                self._debug_info.append(f"\nProcessing file: {filename}")
                loader_class = self._get_loader_for_file(filename)
                
                if not loader_class:
                    self._debug_info.append(f"Skipping {filename} - unsupported format")
                    continue
                
                # Create temp directory if it doesn't exist
                temp_dir = os.path.join(os.path.dirname(__file__), "..", "temp")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Save to temp file with original name
                temp_path = os.path.join(temp_dir, filename)
                self._debug_info.append(f"Saving to temp file: {temp_path}")
                
                try:
                    with open(temp_path, 'wb') as tmp_file:
                        tmp_file.write(file_content)
                        temp_files.append(temp_path)
                    
                    # Load the document
                    self._debug_info.append(f"Loading document with {loader_class.__name__}")
                    loader = loader_class(temp_path)
                    docs = loader.load()
                    self._debug_info.append(f"Successfully loaded {len(docs)} pages/sections")
                    documents.extend(docs)
                except Exception as e:
                    self._debug_info.append(f"Error loading {filename}: {str(e)}")
                        
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
            if not documents:
                return 0, "No valid documents found in uploads"
                
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            
            # Add documents to vector store
            if not texts:
                self._debug_info.append("No valid text chunks extracted")
                return 0, "No valid text chunks extracted from documents"
            
            self._debug_info.append(f"Vector store path: {vector_store_path}")
            self._debug_info.append(f"Processing {len(texts)} text chunks")
            
            # Always create a new vector store from the current documents
            self._debug_info.append("Creating vector store from documents")
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            
            # Save the vector store
            os.makedirs(vector_store_path, exist_ok=True)
            self.vector_store.save_local(vector_store_path)
            
            # Verify save was successful
            if os.path.exists(os.path.join(vector_store_path, "index.faiss")):
                self._debug_info.append(f"Successfully saved vector store with {len(texts)} chunks")
                # Test search to verify
                test_results = self.vector_store.similarity_search(texts[0].page_content[:50], k=1)
                if test_results:
                    self._debug_info.append("Vector store verified with test search")
            else:
                self._debug_info.append("Warning: Vector store save may have failed")
                
            return len(texts), f"Successfully processed {len(documents)} documents into {len(texts)} chunks"
            
        except Exception as e:
            return 0, f"Error processing files: {str(e)}"
        
    def initialize(self, documents_path: str, openai_api_key: str) -> bool:
        self._debug_info = []  # Reset debug info
        try:
            if not openai_api_key:
                self._debug_info.append("OpenAI API key is required")
                return False
            
            # Initialize embeddings with API key
            try:
                self._debug_info.append(f"Initializing OpenAI embeddings with key: {openai_api_key[:6]}...")
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=openai_api_key,
                    model="text-embedding-ada-002"
                )
                # Test the embeddings
                test_embedding = self.embeddings.embed_query("test")
                self._debug_info.append(f"Successfully initialized OpenAI embeddings (vector size: {len(test_embedding)})")
            except Exception as e:
                self._debug_info.append(f"Failed to initialize OpenAI embeddings: {str(e)}")
                return False
            
            # Always try to load existing vector store first
            vector_store_path = self.get_vector_store_path()
            index_path = os.path.join(vector_store_path, "index.faiss")
            pkl_path = os.path.join(vector_store_path, "index.pkl")
            
            self._debug_info.append(f"Vector store path: {vector_store_path}")
            self._debug_info.append(f"Checking for index files:\n- {index_path}\n- {pkl_path}")
            
            if os.path.exists(index_path) and os.path.exists(pkl_path):
                try:
                    self._debug_info.append("Found existing vector store, loading...")
                    self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
                    self._debug_info.append("Successfully loaded vector store")
                    return True
                except Exception as e:
                    self._debug_info.append(f"Failed to load vector store: {str(e)}")
                    # Delete corrupted files
                    try:
                        os.remove(index_path)
                        os.remove(pkl_path)
                        self._debug_info.append("Removed corrupted vector store files")
                    except:
                        pass
            else:
                self._debug_info.append("No existing vector store found")
            
            # Initialize empty vector store
            self._debug_info.append("Initializing new vector store")
            os.makedirs(vector_store_path, exist_ok=True)
            self.vector_store = None  # Ensure it's empty
            return True
            
            return True
            
        except Exception as e:
            print(f"Error initializing RAG module: {str(e)}")
            return False
            
    def format_search_results(self, query: str, docs: List[Document], timestamp: str) -> str:
        """Format search results with metadata"""
        formatted_results = [
            f"Search Query: {query}",
            f"Timestamp: {timestamp}",
            f"Number of Results: {len(docs)}",
            "\n=== Search Results ==="
        ]
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get('source', 'Unknown source')
            formatted_results.append(
                f"\nDocument {i}:\n" \
                f"Source: {source}\n" \
                f"Content:\n{content}\n" \
                f"{'-' * 80}"
            )
        
        return "\n".join(formatted_results)
    
    def save_search_results(self, formatted_results: str, timestamp: str) -> str:
        """Save search results to a file"""
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename with timestamp
            filename = f"search_results_{timestamp.replace(':', '-')}.txt"
            filepath = os.path.join(output_dir, filename)
            
            # Save results
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(formatted_results)
            
            self._debug_info.append(f"Results saved to: {filepath}")
            return filepath
        except Exception as e:
            self._debug_info.append(f"Error saving results: {str(e)}")
            return ""
    
    def search(self, query: str, k: int = 3) -> str:
        """Search for documents relevant to the query"""
        self._debug_info = []  # Reset debug info
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        try:
            # Ensure embeddings are initialized
            if not self.embeddings:
                self._debug_info.append("Embeddings not initialized. Checking for API key...")
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    return "Error: OpenAI API key not found. Please set it in your environment variables."
                try:
                    self.embeddings = OpenAIEmbeddings(
                        openai_api_key=api_key,
                        model="text-embedding-ada-002"
                    )
                    self._debug_info.append("Successfully initialized embeddings")
                except Exception as e:
                    self._debug_info.append(f"Failed to initialize embeddings: {str(e)}")
                    return "Error: Could not initialize embeddings. Please check your API key."
            
            # Try to load vector store if not initialized
            if not self.vector_store:
                vector_store_path = self.get_vector_store_path()
                index_path = os.path.join(vector_store_path, "index.faiss")
                self._debug_info.append(f"Checking vector store at: {index_path}")
                
                if os.path.exists(index_path):
                    try:
                        self._debug_info.append("Found vector store, loading...")
                        self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
                        self._debug_info.append("Successfully loaded vector store")
                    except Exception as e:
                        self._debug_info.append(f"Failed to load vector store: {str(e)}")
                        return "Error: Failed to load document store. Please try uploading your documents again."
                else:
                    self._debug_info.append("No vector store found")
                    return "Please upload and process some documents first."
            
            # Perform search
            self._debug_info.append(f"Searching for: {query}")
            docs = self.vector_store.similarity_search(query, k=k)
            self._debug_info.append(f"Found {len(docs)} relevant documents")
            
            if not docs:
                return "No relevant documents found for your query."
            
            # Format and save results
            formatted_results = self.format_search_results(query, docs, timestamp)
            output_file = self.save_search_results(formatted_results, timestamp)
            
            # Format results for display
            display_results = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()
                source = doc.metadata.get('source', 'Unknown source')
                display_results.append(f"**Document {i}** _(from {source})_\n\n{content}\n")
            
            if output_file:
                display_results.append(f"\n💾 _Results saved to: {os.path.basename(output_file)}_")
            
            return "\n---\n".join(display_results)
            
        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            self._debug_info.append(error_msg)
            return error_msg

    def query(self, query: str, k: int = 3) -> Optional[str]:
        """Legacy method - use search() instead"""
        return self.search(query, k)
