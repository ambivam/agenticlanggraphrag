from dotenv import load_dotenv
import os
from pathlib import Path

# Get the absolute path to the .env file
env_path = Path(__file__).resolve().parent / '.env'

# Debug print
print(f"Looking for .env file at: {env_path}")

# Load environment variables with explicit path and override
load_dotenv(dotenv_path=str(env_path), override=True)

# Debug print
print(f"Environment variables loaded: {[k for k in os.environ.keys() if not k.startswith('_')]}")

class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    @classmethod
    def debug_info(cls) -> dict:
        """Get debug information about configuration"""
        return {
            "env_file_exists": env_path.exists(),
            "env_file_path": str(env_path),
            "openai_key_exists": bool(cls.OPENAI_API_KEY),
            "loaded_vars": [k for k in os.environ.keys() if not k.startswith("_")]
        }
    
    # MySQL
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_USER = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
    
    # JIRA
    JIRA_SERVER = os.getenv("JIRA_SERVER")
    JIRA_EMAIL = os.getenv("JIRA_EMAIL")
    JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
    
    # SerpAPI
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    
    # RAG
    RAG_DOCUMENTS_PATH = os.getenv("RAG_DOCUMENTS_PATH", "documents")
    
    @classmethod
    def get_all_config(cls) -> dict:
        """Get all configuration as a dictionary"""
        return {
            "openai_api_key": cls.OPENAI_API_KEY,
            "mysql_host": cls.MYSQL_HOST,
            "mysql_user": cls.MYSQL_USER,
            "mysql_password": cls.MYSQL_PASSWORD,
            "mysql_database": cls.MYSQL_DATABASE,
            "jira_server": cls.JIRA_SERVER,
            "jira_email": cls.JIRA_EMAIL,
            "jira_api_token": cls.JIRA_API_TOKEN,
            "serpapi_api_key": cls.SERPAPI_API_KEY,
            "rag_documents_path": cls.RAG_DOCUMENTS_PATH
        }
