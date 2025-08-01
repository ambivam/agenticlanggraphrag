from typing import Optional, List
from jira import JIRA
import os
from dotenv import load_dotenv
import traceback
import re

load_dotenv()

__all__ = ['jira_config', 'search_jira_issues']

class JiraConfig:
    """Configuration class for JIRA integration."""
    
    def __init__(self):
        # Load configuration from environment variables
        self.server = os.getenv('JIRA_SERVER')
        self.username = os.getenv('JIRA_USERNAME')
        self.api_token = os.getenv('JIRA_API_TOKEN')
        self.project_key = None  # Will be set from UI input
        self.is_enabled = True  # Always enabled if env vars are present
        
        print(f"[DEBUG] Loaded JIRA config from environment:")
        print(f"[DEBUG] Server: {self.server}")
        print(f"[DEBUG] Username: {self.username}")

    def set_project_key(self, project_key: str) -> None:
        """Set the JIRA project key."""
        self.project_key = project_key
        print(f"[DEBUG] Set project key to: {project_key}")

    def is_configured(self) -> bool:
        """Check if all required configuration is present."""
        return all([
            self.server,
            self.username,
            self.api_token,
            self.project_key
        ])

jira_config = JiraConfig()

def get_jira_tool() -> Optional[JIRA]:
    """Initialize and return JIRA client if configuration is valid."""
    print("[DEBUG] Checking JIRA configuration...")
    print(f"[DEBUG] Server: {jira_config.server}")
    print(f"[DEBUG] Username: {jira_config.username}")
    print(f"[DEBUG] Project Key: {jira_config.project_key}")
    print(f"[DEBUG] Is Enabled: {jira_config.is_enabled}")
    
    if not jira_config.is_configured():
        print("[DEBUG] JIRA not configured properly")
        return None
        
    try:
        # Try to create JIRA client
        jira = JIRA(
            server=jira_config.server,
            basic_auth=(jira_config.username, jira_config.api_token)
        )
        
        # Test connection by trying to access the project
        project = jira.project(jira_config.project_key)
        print(f"[DEBUG] Successfully connected to JIRA project: {project.key}")
        
        return jira
    except Exception as e:
        print(f"Error initializing JIRA: {str(e)}")
        print("[DEBUG] Full traceback:")
        traceback.print_exc()
        return None

def extract_jira_keys(text: str, project_key: str) -> List[str]:
    """Extract JIRA issue keys from text."""
    # Pattern matches PROJECT-123 format, allowing for comma/space separation
    pattern = fr'{project_key}-\d+'
    matches = re.findall(pattern, text)
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

def extract_issue_keys(query: str) -> List[str]:
    """Extract issue keys from query."""
    # Pattern matches PROJECT-123 format, allowing for comma/space separation
    pattern = fr'{jira_config.project_key}-\d+'
    matches = re.findall(pattern, query)
    return list(set(matches))  # Remove duplicates

def search_jira_issues(query: str) -> str:
    """Search for JIRA issues by issue key or text search.
    
    Args:
        query: The search query, can be natural language or issue key(s)
        
    Returns:
        Formatted string with issue details or None if no results found
    """
    print(f"[DEBUG] JIRA Search - Query: {query}")
    print(f"[DEBUG] JIRA Config Status: {jira_config.is_configured()}")
    
    if not jira_config.is_configured():
        print("[DEBUG] JIRA not configured")
        return None
        
    try:
        jira = get_jira_tool()
        if not jira:
            print("[DEBUG] Failed to get JIRA tool")
            return None
            
        # Clean and parse query
        clean_query = query.strip()
        issue_keys = extract_issue_keys(clean_query)
        
        # Build JQL query
        if not issue_keys:
            # Text search if no issue keys found
            jql = f'text ~ "{clean_query}" ORDER BY created DESC'
        else:
            # Search by issue keys
            keys_clause = ' OR '.join(f'key = "{key}"' for key in issue_keys)
            jql = f'({keys_clause})'
        
        # Search issues
        issues = jira.search_issues(jql, maxResults=5)
        
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
    
    def invoke(self, input_text):
        return generate_test_cases(input_text)

def get_jira_tools():
    """Get the JIRA tools."""
    try:
        tools = []
        jira_tool = JIRATool()
        test_case_gen = TestCaseGenerator()
        
        if jira_config.is_configured():
            tools.extend([jira_tool, test_case_gen])
            
        return tools
    except Exception as e:
        print(f"Error creating JIRA tools: {str(e)}")
        return []

# Initialize global JIRA config
jira_config = JiraConfig()
