import os
from module_manager import ModuleManager
from modules.rag_module import RAGModule
from modules.mysql_module import MySQLModule
from modules.jira_module import JIRAModule
from tools.serpapi_tool import get_serp_tool
from config import Config

class SearchOrchestrator:
    def __init__(self, output_dir: str = "output"):
        # Initialize module manager
        self.module_manager = ModuleManager(output_dir)
        
        # Initialize modules
        self.rag = RAGModule()
        self.mysql = MySQLModule()
        self.web_search = get_serp_tool()
        self.jira = JIRAModule()
        
    def initialize_modules(self, config: dict):
        """Initialize enabled modules with their configurations"""
        if self.module_manager.is_module_enabled("rag"):
            self.rag.initialize(
                documents_path=config.get("rag_documents_path", "documents"),
                openai_api_key=config.get("openai_api_key")
            )
            
        if self.module_manager.is_module_enabled("mysql"):
            self.mysql.initialize(
                host=config.get("mysql_host", "localhost"),
                user=config.get("mysql_user"),
                password=config.get("mysql_password"),
                database=config.get("mysql_database")
            )
            
        if self.module_manager.is_module_enabled("jira"):
            self.jira.initialize(
                server=config.get("jira_server"),
                email=config.get("jira_email"),
                api_token=config.get("jira_api_token")
            )
            
    def search(self, query: str):
        """Execute search across all enabled modules"""
        
        # RAG search
        if self.module_manager.is_module_enabled("rag"):
            result = self.rag.query(query)
            if result:
                self.module_manager.write_output("rag", result)
                
        # MySQL search
        if self.module_manager.is_module_enabled("mysql"):
            result = self.mysql.execute_query(query)
            if result:
                self.module_manager.write_output("mysql", result)
                
        # Web search
        if self.module_manager.is_module_enabled("web_search"):
            result = self.web_search.run(query)
            if result:
                self.module_manager.write_output("web_search", result)
                
        # JIRA search
        if self.module_manager.is_module_enabled("jira"):
            result = self.jira.search_issues(query)
            if result:
                self.module_manager.write_output("jira", result)
                
    def cleanup(self):
        """Cleanup resources"""
        if self.module_manager.is_module_enabled("mysql"):
            self.mysql.close()

def main():
    # Get configuration from environment
    config = Config.get_all_config()
    
    # Create orchestrator
    orchestrator = SearchOrchestrator()
    
    # Enable desired modules (this would typically come from UI checkboxes)
    orchestrator.module_manager.enable_module("rag")
    orchestrator.module_manager.enable_module("mysql")
    orchestrator.module_manager.enable_module("web_search")
    orchestrator.module_manager.enable_module("jira")
    
    # Initialize modules
    orchestrator.initialize_modules(config)
    
    # Example search
    query = "example search query"
    orchestrator.search(query)
    
    # Cleanup
    orchestrator.cleanup()

if __name__ == "__main__":
    main()
