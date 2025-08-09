from typing import List, Dict, Any
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class JIRAStorage:
    def __init__(self, index_name: str = "faiss_index"):
        print(f"\nInitializing JIRA Storage with index: {index_name}")
        
        # Load environment variables
        env_path = Path(__file__).parent.parent / '.env'
        print(f"Loading .env from: {env_path}")
        load_dotenv(dotenv_path=env_path)
        
        # Check OpenAI API key
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        print("Found OpenAI API key")
        
        # Get absolute path to index
        self.index_name = str(Path(__file__).parent.parent / index_name)
        print(f"Full index path: {self.index_name}")
        
        # Initialize embeddings with explicit key
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        print("Initialized OpenAI embeddings")
        
        self._ensure_index()
    
    def _ensure_index(self):
        """Ensure FAISS index exists, create if not"""
        try:
            index_path = Path(self.index_name)
            print(f"Checking index at: {index_path.absolute()}")
            
            if not index_path.exists():
                print("Index not found, creating new one...")
                # Create directory if it doesn't exist
                index_path.mkdir(parents=True, exist_ok=True)
                # Create empty index
                empty_texts = ["placeholder"]
                FAISS.from_texts(empty_texts, self.embeddings, folder_path=str(index_path))
                print("Created new FAISS index")
            else:
                print("Found existing FAISS index")
        except Exception as e:
            print(f"Error ensuring index: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")            
            raise
    
    def store_jira_results(self, results: List[Dict[str, Any]]) -> int:
        """Store JIRA results in FAISS index
        
        Args:
            results: List of JIRA issues with keys: key, summary, description, status, assignee
            
        Returns:
            int: Number of chunks stored
        """
        print(f"\nStoring {len(results) if results else 0} JIRA results...")
        print(f"Results type: {type(results)}")
        
        if not results:
            print("No results to store")
            return 0
            
        if not isinstance(results, list):
            print(f"Error: results must be a list, got {type(results)}")
            return 0
            
        try:
            # Validate results
            required_keys = ['key', 'summary', 'description', 'status', 'assignee']
            for i, issue in enumerate(results):
                print(f"\nValidating issue {i+1}:")
                print(f"Issue type: {type(issue)}")
                if not isinstance(issue, dict):
                    print(f"Error: issue must be a dict, got {type(issue)}")
                    continue
                    
                print(f"Issue keys: {issue.keys()}")
                missing_keys = [key for key in required_keys if key not in issue]
                if missing_keys:
                    print(f"Error: Missing required keys: {missing_keys}")
                    continue
                print("Issue validation passed")
            
            # Load existing index
            print(f"\nLoading index from: {self.index_name}")
            try:
                db = FAISS.load_local(self.index_name, self.embeddings)
                print("Successfully loaded existing FAISS index")
            except Exception as e:
                print(f"Error loading index: {str(e)}")
                print("Creating new index...")
                db = FAISS.from_texts(["placeholder"], self.embeddings)
                print("Created new index")
            
            # Prepare documents for storage
            texts = []
            metadatas = []
            
            for issue in results:
                if not isinstance(issue, dict):
                    continue
                    
                if not all(key in issue for key in required_keys):
                    continue
                    
                print(f"Processing issue: {issue.get('key', 'unknown')}")
                # Create rich text combining all issue fields
                text = f"""JIRA Issue {issue['key']}
Summary: {issue['summary']}
Status: {issue['status']}
Assignee: {issue['assignee']}
Description:
{issue['description']}
"""
                texts.append(text)
                
                # Store metadata
                metadata = {
                    "source": f"jira/{issue['key']}",
                    "type": "jira_issue",
                    "key": issue["key"],
                    "summary": issue["summary"],
                    "status": issue["status"],
                    "assignee": issue["assignee"]
                }
                metadatas.append(metadata)
            
            print(f"Adding {len(texts)} texts to index...")
            # Add to index
            db.add_texts(texts, metadatas=metadatas)
            
            print("Saving updated index...")
            # Save updated index
            db.save_local(self.index_name)
            print("Successfully saved index")
            
            return len(texts)
            
        except Exception as e:
            print(f"Error storing JIRA results: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return 0
