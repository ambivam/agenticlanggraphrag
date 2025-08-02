from jira import JIRA
from typing import Optional, List, Any
import re
import traceback
import os
from pathlib import Path
from dotenv import load_dotenv

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

class JIRATool:
    def __init__(self):
        self.is_jira_tool = True
    
    def invoke(self, query: str) -> str:
        try:
            print("\nJIRA Tool - Starting search...")
            print(f"Query: {query}")
            
            if not jira_config.is_configured():
                print("JIRA Tool - Not configured")
                return None
                
            # Get JIRA instance
            print("JIRA Tool - Getting JIRA instance...")
            jira = jira_config.get_jira()
            if not jira:
                print("JIRA Tool - Failed to get JIRA instance")
                return None
                
            # Clean and parse query
            clean_query = query.strip()
            issue_keys = extract_issue_keys(clean_query)
            
            # Build JQL query
            if not issue_keys:
                # If input looks like an issue key but didn't match pattern
                if re.match(r'^[A-Za-z]+-\d+$', clean_query):
                    jql = f'key = "{clean_query.upper()}"'
                else:
                    # Text search if no issue keys found
                    jql = f'project = "{jira_config.project_key}" AND text ~ "{clean_query}" ORDER BY created DESC'
            else:
                # Search by issue keys
                keys_clause = ' OR '.join(f'key = "{key}"' for key in issue_keys)
                jql = f'({keys_clause})'
            
            # Debug logging
            print(f"JIRA Search - JQL: {jql}")
            print(f"JIRA Config - Project Key: {jira_config.project_key}")
            
            # Search issues
            issues = jira.search_issues(jql, maxResults=5)
            print(f"JIRA Search - Found {len(issues)} issues")
            
            if not issues:
                return None
                
            # Format results
            results = []
            for issue in issues:
                # Get assignee info if available
                assignee = getattr(issue.fields, 'assignee', None)
                assignee_name = assignee.displayName if assignee else 'Unassigned'
                
                # Format the issue details
                results.append(
                    f"### {issue.key}: {issue.fields.summary}\n"
                    f"**Status:** {issue.fields.status.name}  |  **Assignee:** {assignee_name}\n"
                    f"**Description:**\n{issue.fields.description or 'No description'}\n"
                )
            
            # Format response
            if len(issues) > 1:
                return f"Found {len(issues)} JIRA issues:\n\n" + "\n\n".join(results)
            else:
                return "\n".join(results)
        except Exception as e:
            print(f"Error searching JIRA: {str(e)}")
            return None

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
    
    # Create JIRA tool
    try:
        print("\nCreating JIRA tool...")
        jira_tool = JIRATool()
        print("JIRA tool created")
        tools.append(jira_tool)
    except Exception as e:
        print(f"\nError creating JIRA tool:\n{str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return []
    
    # Create test case generator if OpenAI key exists
    if os.getenv('OPENAI_API_KEY'):
        try:
            print("\nCreating test case generator...")
            test_case_gen = TestCaseGenerator()
            print("Test case generator created")
            tools.append(test_case_gen)
        except Exception as e:
            print(f"\nError creating test case generator:\n{str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            print("Continuing with JIRA tool only...")
    else:
        print("\nSkipping test case generator - OpenAI API key not found")
        print("JIRA search will work, but test case generation is disabled")
    
    print(f"\nJIRA tools found: {len(tools)}")
    return tools
