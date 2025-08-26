from jira import JIRA
from typing import TypedDict, Optional, List, Dict, Any, Callable
from datetime import datetime
from jira import JIRA
import os
import json
import traceback
import re
import requests
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
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
        description="JQL query parts",
        examples=[
            "priority = High",
            "type = Bug",
            "assignee = 'John'"
        ],
        default_factory=list
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
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4.1-nano"
        )
        self.embeddings = OpenAIEmbeddings()
        # self.text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,
        #     chunk_overlap=100,
        #     separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        # )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Ensure FAISS directory exists
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.faiss_dir = os.path.join(base_dir, "faiss_index")
        os.makedirs(self.faiss_dir, exist_ok=True)
        
        # Setup LangChain query parsing
        self.query_parser = PydanticOutputParser(pydantic_object=JIRAQueryParser)
        self.query_prompt = PromptTemplate(
            template="""Convert the following natural language JIRA query into a structured format.
Only include search and filter operations, no create/update/delete operations.

Rules:
1. Convert natural language to proper JIRA JQL syntax
2. Extract any text search terms
3. Identify filters like priority, status, type
4. Handle special cases like 'all bugs', 'high priority issues'

Query: {query}

{format_instructions}
""",
            input_variables=["query"],
            partial_variables={"format_instructions": self.query_parser.get_format_instructions()}
        )
        self.query_chain = LLMChain(
            llm=self.llm,
            prompt=self.query_prompt,
            output_parser=self.query_parser,
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
            # Check if query is a direct issue key
            query = state["query"]
            if query.upper().startswith("ES-") and query[3:].isdigit():
                query_info = {
                    "query_type": "issue",
                    "jql_parts": [f'issuekey = {query.upper()}'],
                    "filters": {}
                }
            # Handle common patterns
            elif "high priority" in query.lower():
                query_info = {
                    "query_type": "search",
                    "jql_parts": ["priority = High"],
                    "filters": {}
                }
            elif "bugs" in query.lower() or "bug" in query.lower():
                query_info = {
                    "query_type": "search",
                    "jql_parts": ["type = Bug"],
                    "filters": {}
                }
            else:
                # Use LangChain parsing for complex queries
                query_result = self.query_chain.run(query=query)
                query_info = {
                    "query_type": query_result.query_type,
                    "jql_parts": query_result.jql_parts,
                    "filters": query_result.filters
                }
            
            print(f"Query Info: {query_info}")
            return {
                "query": state["query"],
                "query_info": query_info
            }
            
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
            
            # Add any JQL parts from query info
            if query_info["jql_parts"]:
                jql_parts.extend(query_info["jql_parts"])
            
            # Add text search if specified
            if "text" in query_info["filters"]:
                jql_parts.append(f'text ~ "{query_info["filters"]["text"]}"')
            
            # Build final JQL
            jql = " AND ".join(jql_parts) if jql_parts else ""
            print(f"JQL: {jql}")
            
            return {
                "query": state["query"],
                "query_info": query_info,
                "jql": jql
            }
            
        except Exception as e:
            print(f"Error building JQL: {str(e)}")
            return {"query": state["query"], "error": f"Failed to build JQL query: {str(e)}"}
    
    def _execute_jira_query(self, state: JIRAState) -> JIRAState:
        """Execute JQL query and format results"""
        try:
            jira = jira_config.get_jira()
            if not jira:
                return {"query": state["query"], "error": "JIRA is not configured"}
            
            print(f"Executing JQL: {state['jql']}")
            
            # For direct issue lookup, use issue() method
            if state["query_info"]["query_type"] == "issue":
                issue_key = state["query"].upper()
                try:
                    issue = jira.issue(issue_key)
                    issues = [issue]
                except Exception as e:
                    print(f"Issue not found: {issue_key}")
                    return {"query": state["query"], "error": f"Issue {issue_key} not found"}
            else:
                # For searches, use search_issues()
                issues = jira.search_issues(
                    state["jql"] or "project = \"" + jira_config.project_key + "\"",
                    maxResults=30
                )
            
            results = []
            for issue in issues:
                description = issue.fields.description or ""
                
                # Extract and fetch link content if description exists
                link_content = ""
                if description:
                    link_content = self._extract_and_fetch_link_content(description)
                
                issue_dict = {
                    "key": issue.key,
                    "summary": issue.fields.summary,
                    "status": str(issue.fields.status),
                    "created": str(issue.fields.created),
                    "updated": str(issue.fields.updated),
                    "description": description,
                    "link_content": link_content
                }
                results.append(issue_dict)
                
                # Store in FAISS with link content
                self._store_in_faiss(issue_dict)
            
            return {
                "query": state["query"],
                "query_info": state["query_info"],
                "jql": state["jql"],
                "results": results
            }
            
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            traceback.print_exc()
            return {"query": state["query"], "error": f"Failed to execute query: {str(e)}"}
            
    def _extract_and_fetch_link_content(self, description: str) -> str:
        """Extract URLs from description and fetch their content."""
        try:
            urls = []
            
            # Pattern 1: Atlassian smart-link format [URL|URL|smart-link]
            smart_link_pattern = r'\[([^|]+)\|[^|]+\|smart-link\]'
            smart_links = re.findall(smart_link_pattern, description)
            urls.extend(smart_links)
            
            # Pattern 2: Standard HTTP/HTTPS URLs
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            standard_urls = re.findall(url_pattern, description)
            urls.extend(standard_urls)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            if not unique_urls:
                return ""
            
            link_contents = []
            for url in unique_urls[:3]:  # Limit to first 3 URLs to avoid excessive requests
                try:
                    print(f"Fetching content from: {url}")
                    
                    # Check if this is a Confluence URL and use API if possible
                    if 'atlassian.net/wiki' in url or 'confluence' in url:
                        content = self._fetch_confluence_content(url)
                    else:
                        content = self._fetch_url_content(url)
                    
                    if content:
                        link_contents.append(f"\n--- Content from {url} ---\n{content}")
                except Exception as e:
                    print(f"Error fetching content from {url}: {str(e)}")
                    link_contents.append(f"\n--- Error fetching {url} ---\nUnable to retrieve content: {str(e)}")
            
            return "\n".join(link_contents) if link_contents else ""
            
        except Exception as e:
            print(f"Error extracting links: {str(e)}")
            return ""
    
    def _fetch_confluence_content(self, url: str) -> str:
        """Fetch content from Confluence using API when possible."""
        try:
            # First, try to resolve the short URL to get the actual page ID
            actual_page_id = self._resolve_confluence_short_url(url)
            
            if actual_page_id and jira_config.url and jira_config.username and jira_config.api_token:
                # Try to use Confluence API with resolved page ID
                base_url = jira_config.url.replace('/jira', '').rstrip('/')
                if '/wiki' not in base_url:
                    confluence_base = f"{base_url}/wiki"
                else:
                    confluence_base = base_url
                api_url = f"{confluence_base}/rest/api/content/{actual_page_id}?expand=body.storage,space,version"
                
                auth = (jira_config.username, jira_config.api_token)
                headers = {'Accept': 'application/json'}
                
                print(f"Attempting Confluence API call to: {api_url}")
                response = requests.get(api_url, auth=auth, headers=headers, timeout=10)
                
                print(f"Confluence API response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    title = data.get('title', 'Unknown Page')
                    space_name = data.get('space', {}).get('name', 'Unknown Space')
                    
                    # Extract content from storage format
                    storage_content = data.get('body', {}).get('storage', {}).get('value', '')
                    
                    if storage_content:
                        # Parse HTML content from storage format
                        soup = BeautifulSoup(storage_content, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Get text content
                        text = soup.get_text()
                        
                        # Clean up text
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        clean_text = ' '.join(chunk for chunk in chunks if chunk)
                        
                        # Limit content length
                        if len(clean_text) > 8000:
                            clean_text = clean_text[:8000] + "... [Content truncated]"
                        
                        return f"Confluence Page: {title} (Space: {space_name})\n\n{clean_text}"
                    else:
                        return f"Confluence Page: {title} (Space: {space_name})\n\nNo content available"
                else:
                    print(f"Confluence API failed with status {response.status_code}")
                    if response.status_code == 404:
                        return f"Confluence page not found or access denied: {url}"
                    elif response.status_code == 401:
                        return f"Authentication failed for Confluence page: {url}"
                    else:
                        print(f"Error response: {response.text}")
            
            # If API fails, try authenticated web scraping
            return self._fetch_confluence_web_content(url)
            
        except Exception as e:
            print(f"Error fetching Confluence content: {str(e)}")
            return f"Error accessing Confluence page: {str(e)}"
    
    def _resolve_confluence_short_url(self, url: str) -> Optional[str]:
        """Resolve Confluence short URL to get actual page ID by following redirects."""
        try:
            # Extract short ID from /wiki/x/SHORT_ID format
            wiki_x_pattern = r'/wiki/x/([A-Za-z0-9]+)'
            match = re.search(wiki_x_pattern, url)
            if not match:
                return None
            
            short_id = match.group(1)
            print(f"Attempting to resolve short URL with ID: {short_id}")
            
            if jira_config.url and jira_config.username and jira_config.api_token:
                base_url = jira_config.url.replace('/jira', '').rstrip('/')
                confluence_base = f"{base_url}/wiki"
                auth = (jira_config.username, jira_config.api_token)
                headers = {'Accept': 'application/json'}
                
                # Method 1: Try to follow the redirect to get the actual page URL
                try:
                    print(f"Method 1: Following redirect for {url}")
                    redirect_response = requests.get(url, auth=auth, allow_redirects=True, timeout=10)
                    final_url = redirect_response.url
                    print(f"Final URL after redirect: {final_url}")
                    
                    # Extract page ID from the final URL
                    # Pattern: /wiki/spaces/SPACE/pages/PAGE_ID/Page+Title
                    page_id_pattern = r'/wiki/spaces/[^/]+/pages/(\d+)/'
                    page_match = re.search(page_id_pattern, final_url)
                    if page_match:
                        page_id = page_match.group(1)
                        print(f"Extracted page ID from redirect: {page_id}")
                        return page_id
                except Exception as e:
                    print(f"Method 1 failed: {str(e)}")
                
                # Method 2: Search using CQL with various patterns
                search_queries = [
                    f"id={short_id}",
                    f"key={short_id}",
                    f"shortlink={short_id}"
                ]
                
                for cql in search_queries:
                    try:
                        print(f"Method 2: Searching with CQL: {cql}")
                        search_url = f"{confluence_base}/rest/api/content/search?cql={cql}"
                        response = requests.get(search_url, auth=auth, headers=headers, timeout=10)
                        
                        if response.status_code == 200:
                            data = response.json()
                            results = data.get('results', [])
                            if results:
                                page_id = results[0].get('id')
                                print(f"Found page ID via search: {page_id}")
                                return page_id
                    except Exception as e:
                        print(f"Search with CQL '{cql}' failed: {str(e)}")
                
                # Method 3: Try to get all content and search by title
                try:
                    print("Method 3: Searching all content")
                    all_content_url = f"{confluence_base}/rest/api/content?limit=100"
                    response = requests.get(all_content_url, auth=auth, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get('results', [])
                        # This is a fallback - in practice, we'd need more specific search
                        if results:
                            print(f"Found {len(results)} pages, using first as fallback")
                            return results[0].get('id')
                except Exception as e:
                    print(f"Method 3 failed: {str(e)}")
            
            print(f"All resolution methods failed, using short_id as fallback: {short_id}")
            return short_id  # Fallback to using short_id as page_id
            
        except Exception as e:
            print(f"Error resolving short URL: {str(e)}")
            return None
    
    def _fetch_confluence_web_content(self, url: str) -> str:
        """Fetch Confluence content using authenticated web scraping."""
        try:
            print(f"Attempting authenticated web scraping of: {url}")
            
            # Use session with authentication
            session = requests.Session()
            
            # Set headers to mimic browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            session.headers.update(headers)
            
            # Try with basic auth
            if jira_config.username and jira_config.api_token:
                session.auth = (jira_config.username, jira_config.api_token)
            
            # Follow redirects to get the actual page
            response = session.get(url, timeout=15, verify=False, allow_redirects=True)
            print(f"Web scraping response status: {response.status_code}")
            print(f"Final URL after redirects: {response.url}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Try multiple selectors for Confluence content
                content_selectors = [
                    {'id': 'main-content'},
                    {'class': 'wiki-content'},
                    {'class': 'page-content'},
                    {'id': 'content'},
                    {'class': 'confluence-content'},
                    {'class': 'aui-page-panel-content'}
                ]
                
                content_div = None
                for selector in content_selectors:
                    content_div = soup.find('div', selector)
                    if content_div:
                        print(f"Found content using selector: {selector}")
                        break
                
                if content_div:
                    # Remove unwanted elements
                    for unwanted in content_div(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                        unwanted.decompose()
                    
                    # Try to get the page title
                    title_elem = soup.find('title') or soup.find('h1')
                    title = title_elem.get_text().strip() if title_elem else "Unknown Page"
                    
                    text = content_div.get_text()
                else:
                    # Fallback: try to extract from body
                    body = soup.find('body')
                    if body:
                        # Remove unwanted elements
                        for unwanted in body(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                            unwanted.decompose()
                        text = body.get_text()
                        title = "Confluence Page"
                    else:
                        text = soup.get_text()
                        title = "Confluence Page"
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Remove common Confluence UI text
                ui_text_to_remove = [
                    "Skip to main content",
                    "Atlassian",
                    "Log in",
                    "Sign up",
                    "Dashboard",
                    "People",
                    "Apps",
                    "Create",
                    "Help"
                ]
                
                for ui_text in ui_text_to_remove:
                    clean_text = clean_text.replace(ui_text, "")
                
                # Limit content length
                if len(clean_text) > 8000:
                    clean_text = clean_text[:8000] + "... [Content truncated]"
                
                if clean_text.strip():
                    return f"Confluence Page: {title}\n\n{clean_text}"
                else:
                    return f"Confluence Page: {title}\n\nNo readable content found"
                    
            elif response.status_code == 401:
                return "Authentication failed - please check JIRA credentials"
            elif response.status_code == 403:
                return "Access forbidden - insufficient permissions for this page"
            elif response.status_code == 404:
                return "Page not found - the page may have been moved or deleted"
            else:
                return f"Unable to access page (Status: {response.status_code})"
                
        except Exception as e:
            print(f"Web scraping error: {str(e)}")
            return f"Error accessing page: {str(e)}"
    
    def _fetch_url_content(self, url: str) -> str:
        """Fetch and extract text content from a URL."""
        try:
            # Set headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Make request with timeout
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit content length
            if len(text) > 8000:
                text = text[:8000] + "... [Content truncated]"
            
            return text
            
        except requests.RequestException as e:
            return f"Request error: {str(e)}"
        except Exception as e:
            return f"Parsing error: {str(e)}"
    
    def _store_in_faiss(self, issue_dict):
        """Store JIRA issue in FAISS index."""
        try:
            # Create document for vectorization including link content
            content = f"JIRA Issue {issue_dict['key']}:\n" \
                     f"Summary: {issue_dict['summary']}\n" \
                     f"Status: {issue_dict['status']}\n" \
                     f"Created: {issue_dict['created']}\n" \
                     f"Updated: {issue_dict['updated']}\n" \
                     f"Description: {issue_dict['description']}"
            
            # Add link content if available
            if issue_dict.get('link_content'):
                content += f"\n\nLinked Content: {issue_dict['link_content']}"
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "key": issue_dict['key'],
                    "summary": issue_dict['summary'],
                    "status": issue_dict['status'],
                    "created": issue_dict['created'],
                    "updated": issue_dict['updated'],
                    "description": issue_dict['description'],
                    "link_content": issue_dict.get('link_content', '')
                }
            )
            
            # Create chunks with metadata preserved
            chunks = self.text_splitter.split_documents([doc])
            
            # Load existing index if it exists
            if os.path.exists(self.faiss_dir):
                print(f"Loading existing FAISS index from {self.faiss_dir}")
                db = FAISS.load_local(self.faiss_dir, self.embeddings, allow_dangerous_deserialization=True)
                
                # Check if issue already exists and remove old chunks
                existing_docs = db.similarity_search(f"key:{issue_dict['key']}", k=10)
                if existing_docs:
                    print(f"Removing existing chunks for issue {issue_dict['key']}")
                    db._index = None  # Force reindex
                
                # Add new chunks
                db.add_documents(chunks)
            else:
                # Create new index
                db = FAISS.from_documents(chunks, self.embeddings)
            
            # Save updated index
            db.save_local(self.faiss_dir)
            print(f"Stored {len(chunks)} chunks in FAISS for issue {issue_dict['key']}")
            
        except Exception as e:
            print(f"Error storing in FAISS: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
    
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
            
            # Check if this is a direct issue lookup
            if query.upper().startswith("ES-"):
                issue_key = query.upper()
                # Try FAISS first
                try:
                    if os.path.exists(self.faiss_dir):
                        db = FAISS.load_local(self.faiss_dir, self.embeddings, allow_dangerous_deserialization=True)
                        docs = db.similarity_search(f"key:{issue_key}", k=1)
                        if docs and docs[0].metadata.get('key') == issue_key:
                            issue = docs[0].metadata
                            return (
                                f"### {issue['key']}: {issue['summary']}\n"
                                f"**Status:** {issue['status']}\n"
                                f"**Created:** {issue['created']}\n"
                                f"**Updated:** {issue['updated']}\n"
                                f"**Description:**\n{issue['description']}"
                            )
                except Exception as e:
                    print(f"FAISS lookup failed: {str(e)}")
            
            if not jira_config.is_configured():
                print("JIRA MCP Tool - Not configured")
                return "Error: JIRA is not configured"
            
            # Execute the processing chain
            result = self.chain({"query": query})
            
            if "error" in result:
                return f"Error: {result['error']}"
            
            if not result.get("results"):
                return "No matching issues found"
            
            # Format results
            formatted_results = []
            for issue in result["results"]:
                formatted_issue = (
                    f"### {issue['key']}: {issue['summary']}\n"
                    f"**Status:** {issue['status']}\n"
                    f"**Created:** {issue['created']}\n"
                    f"**Updated:** {issue['updated']}\n"
                    f"**Description:**\n{issue['description']}\n"
                )
                
                # Add link content if available
                if issue.get('link_content'):
                    formatted_issue += f"\n**Linked Content:**{issue['link_content']}\n"
                
                formatted_results.append(formatted_issue)
            
            # Format response
            if len(formatted_results) > 1:
                return f"Found {len(formatted_results)} JIRA issues:\n\n" + "\n\n".join(formatted_results)
            else:
                return "\n".join(formatted_results)
                
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
