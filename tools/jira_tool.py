from jira import JIRA
from typing import Optional, List, Any, Dict, TypedDict, Callable
import re
import traceback
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from pydantic import BaseModel, Field

from .test_case_generator import TestCaseGenerator

# Get absolute path to .env file
env_path = Path(__file__).parent.parent / '.env'
print(f"JIRA Tool - Loading .env from: {env_path}")
load_dotenv(dotenv_path=env_path)

__all__ = ['jira_config', 'get_jira_tools']

class JiraConfig:
    """Configuration class for JIRA integration."""
    
    def __init__(self):
        self.reload_config()
    
    def reload_config(self):
        """Reload configuration from environment variables."""
        print("\nReloading JIRA config from environment:")
        
        # Print raw environment variables
        print("\nRaw environment variables:")
        for key in ['JIRA_URL', 'JIRA_USERNAME', 'JIRA_API_TOKEN', 'JIRA_PROJECT_KEY']:
            print(f"{key}: {os.environ.get(key)}")
        
        # Get path to .env file
        env_path = Path(__file__).parent.parent / '.env'
        print(f"\nChecking .env at: {env_path}")
        
        if env_path.exists():
            print("Found .env file")
            with open(env_path) as f:
                print("\n.env contents:")
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        print(line.strip())
            
            # Force reload from environment
            print("\nForce reloading environment...")
            load_dotenv(dotenv_path=env_path, override=True)
            
            # Load configuration
            self.url = os.getenv('JIRA_URL')
            self.username = os.getenv('JIRA_USERNAME')
            self.api_token = os.getenv('JIRA_API_TOKEN')
            self.project_key = os.getenv('JIRA_PROJECT_KEY')
            
            print("\nLoaded configuration:")
            print(f"URL: {self.url}")
            print(f"Username: {self.username}")
            print(f"Project Key: {self.project_key}")
        else:
            print(".env file not found!")
            print(f"Working directory: {os.getcwd()}")
            print("Available files:")
            for f in Path(os.getcwd()).iterdir():
                print(f"  {f.name}")
    
    def is_configured(self) -> bool:
        """Check if JIRA is configured."""
        # Reload config first
        self.reload_config()
        
        print(f"\nJIRA Configuration Check:")
        print(f"URL: {bool(self.url)} ({self.url})")
        print(f"Username: {bool(self.username)} ({self.username})")
        print(f"API Token: {bool(self.api_token)} ({'*' * len(self.api_token) if self.api_token else 'None'})")
        print(f"Project Key: {bool(self.project_key)} ({self.project_key})")
        
        is_valid = all([self.url, self.username, self.api_token, self.project_key])
        print(f"All configured: {is_valid}")
        return is_valid

    def set_project_key(self, project_key: str) -> None:
        """Set the JIRA project key.
        
        Args:
            project_key: The JIRA project key to use
        """
        self.project_key = project_key
    
    def get_jira(self) -> Optional[JIRA]:
        """Get JIRA instance."""
        try:
            print("\nJIRA Config - Getting JIRA instance...")
            if not self.is_configured():
                print("JIRA Config - Not configured")
                return None
                
            print(f"JIRA Config - URL: {self.url}")
            print(f"JIRA Config - Username: {self.username}")
            print(f"JIRA Config - Project Key: {self.project_key}")
            print(f"JIRA Config - API Token: {'*' * (len(self.api_token) if self.api_token else 0)}")
            
            if not all([self.url, self.username, self.api_token, self.project_key]):
                print("JIRA Config - Missing required configuration:")
                if not self.url: print("- Missing URL")
                if not self.username: print("- Missing username")
                if not self.api_token: print("- Missing API token")
                if not self.project_key: print("- Missing project key")
                return None
            
            options = {
                'server': self.url
            }
            
            auth = (self.username, self.api_token)
            print("JIRA Config - Creating JIRA instance...")
            
            jira = JIRA(options, basic_auth=auth)
            print("JIRA Config - JIRA instance created successfully")
            return jira
        except Exception as e:
            print(f"\nError connecting to JIRA:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}")
            return None

jira_config = JiraConfig()

def extract_issue_keys(query: str) -> List[str]:
    """Extract issue keys from query."""
    # Pattern matches PROJECT-123 format
    pattern = r'[A-Z]+-\d+'
    matches = re.findall(pattern, query)
    return list(set(matches))  # Remove duplicates

def clean_natural_language(query: str) -> str:
    """Clean natural language query by removing common JIRA-related phrases."""
    phrases_to_remove = [
        'jira issue',
        'jira ticket',
        'please specify info about',
        'information about',
        'tell me about',
        'what is',
        'show me',
        'find',
        'search for',
    ]
    
    # Case-insensitive removal of phrases
    clean_query = query.lower()
    for phrase in phrases_to_remove:
        clean_query = clean_query.replace(phrase.lower(), '')
    
    # Remove extra whitespace and commas
    clean_query = re.sub(r'\s+', ' ', clean_query)
    clean_query = clean_query.strip(' ,')
    
    return clean_query

class JIRAQueryParser(BaseModel):
    """Parser for JIRA natural language queries"""
    query_type: str = Field(
        description="Type of query",
        examples=["search", "filter", "status"]
    )
    jql_parts: List[str] = Field(
        description="Parts of the JQL query in JIRA syntax",
        examples=[
            ["priority = High"],
            ["type = Bug", "status = Open"],
            ["assignee = currentUser()", "status != Closed"]
        ]
    )
    filters: Dict[str, str] = Field(
        description="Additional field filters to apply",
        examples=[
            {"text": "performance issue"},
            {"summary": "login screen"},
            {"description": "database error"}
        ],
        default_factory=dict
    )

class JIRAState(TypedDict, total=False):
    """State type for JIRA query processing"""
    query: str
    query_info: Dict[str, Any]
    jql: str
    results: List[Dict[str, Any]]
    error: str

class JIRAMCPTool:
    def __init__(self):
        self.is_jira_tool = True
        
        # Initialize LangChain components
        self.query_parser = PydanticOutputParser(pydantic_object=JIRAQueryParser)
        self.llm = OpenAI(temperature=0)
        
        # Define the query parsing prompt
        self.query_prompt = PromptTemplate(
            template="""Convert the following natural language JIRA query into a structured format.
Only include search and filter operations, no create/update/delete operations.

Rules:
1. Convert natural language to proper JIRA JQL syntax
2. Use exact JIRA field names and values (e.g., 'priority = High', 'type = Bug')
3. Don't include any destructive operations
4. If unsure about a field value, use a text search filter

Example 1:
Query: Show me all high priority issues
Response:
{{
    "query_type": "search",
    "jql_parts": ["priority = High"],
    "filters": {{}}
}}

Example 2:
Query: Find bugs assigned to John with performance in the description
Response:
{{
    "query_type": "search",
    "jql_parts": ["type = Bug", "assignee = 'John'"],
    "filters": {{"description": "performance"}}
}}

Example 3:
Query: What's the status of the login page tasks?
Response:
{{
    "query_type": "status",
    "jql_parts": ["type = Task"],
    "filters": {{"summary": "login page"}}
}}

Now convert this query:
Query: {query}

{format_instructions}
""",
            input_variables=["query"],
            partial_variables={"format_instructions": self.query_parser.get_format_instructions()}
        )
        
        # Create LangChain for query parsing
        self.query_chain = LLMChain(
            llm=self.llm,
            prompt=self.query_prompt,
            output_key="text",
            verbose=True
        )
        
        # Setup state processing functions
        self.workflow = {
            "parse_query": self._parse_natural_query,
            "build_jql": self._build_jql_query,
            "execute_query": self._execute_jira_query
        }
        
        # Define the processing chain
        self.chain = self._create_chain()
    
    def _parse_natural_query(self, state: JIRAState) -> JIRAState:
        """Parse natural language query into structured format using LangChain"""
        try:
            # Handle specific query patterns
            query_lower = state["query"].lower()
            if "high priority" in query_lower:
                query_info = {
                    "query_type": "search",
                    "jql_parts": ["priority = High"],
                    "filters": {}
                }
            elif "priority" in query_lower and "description" in query_lower:
                query_info = {
                    "query_type": "search",
                    "jql_parts": [],
                    "filters": {"text": ""}
                }
            else:
                # Use LangChain to parse the query
                chain_response = self.query_chain.invoke({"query": state["query"]})
                response = chain_response["text"]
                print(f"LangChain Response: {response}")
                
                # Parse the response into our expected format
                parsed = self.query_parser.parse(response)
                query_info = parsed.dict()
            
            print(f"Query Info: {query_info}")
            return {
                "query": state["query"],
                "query_info": query_info
            }
        except Exception as e:
            print(f"Error parsing query: {str(e)}")
            traceback.print_exc()
            # Fallback to basic text search
            return {
                "query": state["query"],
                "query_info": {
                    "query_type": "search",
                    "jql_parts": [f'text ~ "{state["query"]}"'],
                    "filters": {}
                }
            }
    
    def _build_jql_query(self, state: JIRAState) -> JIRAState:
        """Build JQL query from parsed information"""
        try:
            query_info = state["query_info"]
            jql_parts = []
            
            # Always restrict to project
            if jira_config.project_key:
                jql_parts.append(f'project = "{jira_config.project_key}"')
            
            # Add query parts from the parsed info
            if query_info["jql_parts"]:
                jql_parts.extend(query_info["jql_parts"])
            
            # Special handling for priority and description query
            if "priority" in state["query"].lower() and "description" in state["query"].lower():
                jql_parts.append("priority IS NOT EMPTY")
            
            # Add filters
            for field, value in query_info["filters"].items():
                if value:  # Only add if value is not empty
                    jql_parts.append(f'{field} ~ "{value}"')
            
            # Add ORDER BY if not present
            jql = " AND ".join(jql_parts)
            if "ORDER BY" not in jql:
                jql += " ORDER BY created DESC"
                
            print(f"Generated JQL: {jql}")
            return {"query": state["query"], "query_info": query_info, "jql": jql}
        except Exception as e:
            print(f"Error building JQL: {str(e)}")
            return {"query": state["query"], "error": f"Failed to build JQL query: {str(e)}"}
    
    def _execute_jira_query(self, state: JIRAState) -> JIRAState:
        """Execute the JQL query against JIRA"""
        try:
            print("\nExecuting JIRA query...")
            jira = jira_config.get_jira()
            if not jira:
                print("JIRA not configured")
                return {"query": state["query"], "error": "JIRA not configured"}
            
            print(f"JQL Query: {state['jql']}")
            issues = jira.search_issues(state["jql"], maxResults=10)
            print(f"Found {len(issues)} issues")
            
            results = []
            for issue in issues:
                print(f"Processing issue: {issue.key}")
                assignee = getattr(issue.fields, 'assignee', None)
                assignee_name = assignee.displayName if assignee else 'Unassigned'
                
                issue_data = {
                    "key": issue.key,
                    "summary": issue.fields.summary,
                    "status": issue.fields.status.name,
                    "assignee": assignee_name,
                    "description": issue.fields.description or 'No description'
                }
                print(f"Issue data: {issue_data['key']} - {issue_data['summary']}")
                results.append(issue_data)
            
            print(f"Processed {len(results)} issues successfully")
            return {
                "query": state["query"],
                "query_info": state["query_info"],
                "jql": state["jql"],
                "results": results
            }
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()})")
            return {"query": state["query"], "error": str(e)}
    
    def _create_chain(self) -> Callable:
        """Create a processing chain from workflow steps"""
        def chain(state: JIRAState) -> JIRAState:
            # Execute each step in sequence
            try:
                state = self.workflow["parse_query"](state)
                if "error" in state:
                    return state
                    
                state = self.workflow["build_jql"](state)
                if "error" in state:
                    return state
                    
                state = self.workflow["execute_query"](state)
                return state
            except Exception as e:
                return {"query": state["query"], "error": str(e)}
        return chain
    
    def invoke(self, query: str) -> str:
        try:
            print("\nJIRA MCP Tool - Starting search...")
            print(f"Query: {query}")
            
            if not jira_config.is_configured():
                print("JIRA MCP Tool - Not configured")
                return "Error: JIRA is not configured"
            
            # Execute the processing chain
            result = self.chain({"query": query})
            
            print("\nJIRA Query Results:")
            print(f"Result type: {type(result)}")
            print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            
            if "error" in result:
                return f"Error: {result['error']}"
            
            if not result.get("results"):
                print("No results found in query response")
                return "No matching issues found"
            
            print(f"Number of JIRA results: {len(result['results'])}")
            print("First result sample:")
            if result['results']:
                first_result = result['results'][0]
                print(f"Keys in result: {first_result.keys()}")
                print(f"Key: {first_result.get('key')}")
                print(f"Summary: {first_result.get('summary')}")
            
            # Store results in FAISS
            try:
                print("\nAttempting to store in FAISS...")
                from .jira_storage import JIRAStorage
                storage = JIRAStorage()
                print("Created storage instance")
                num_stored = storage.store_jira_results(result["results"])
                print(f"Successfully stored {num_stored} JIRA issues")
            except Exception as e:
                print(f"Error storing JIRA results: {str(e)}")
                import traceback
                print(f"Storage error traceback:\n{traceback.format_exc()}")
                num_stored = 0
            
            # Format results
            formatted_results = []
            for issue in result["results"]:
                formatted_results.append(
                    f"### {issue['key']}: {issue['summary']}\n"
                    f"**Status:** {issue['status']}  |  **Assignee:** {issue['assignee']}\n"
                    f"**Description:**\n{issue['description']}\n"
                )
            
            # Format response
            response = []
            if len(formatted_results) > 1:
                response.append(f"Found {len(formatted_results)} JIRA issues:\n\n" + "\n\n".join(formatted_results))
            else:
                response.append("\n".join(formatted_results))
                
            # Add storage confirmation
            response.append(f"\n\n> 💾 Stored {num_stored} JIRA issues in knowledge base for future reference.")
            
            return "\n".join(response)
                
        except Exception as e:
            print(f"Error in JIRA MCP Tool: {str(e)}")
            return f"Error: {str(e)}"

class TestCaseGenerator:
    def __init__(self):
        self.is_test_case_generator = True
        # Initialize test case generator
        from .test_case_generator import TestCaseGenerator as TCG
        self.generator = TCG()
    
    def invoke(self, input_text: str) -> str:
        """Generate test cases for a JIRA issue.
        
        Args:
            input_text: JIRA issue content
            
        Returns:
            str: Generated test cases in Gherkin format
        """
        return self.generator.invoke(input_text)

def get_jira_tools() -> List[Any]:
    """Get JIRA tools."""
    tools = []
    
    # Create JIRA MCP tool
    try:
        print("\nCreating JIRA MCP tool...")
        jira_tool = JIRAMCPTool()
        print("JIRA MCP tool created")
        tools.append(jira_tool)
    except Exception as e:
        print(f"\nError creating JIRA tool:\n{str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return []
    
    print(f"\nJIRA tools found: {len(tools)}")
    return tools
