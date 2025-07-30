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
        self.server = None
        self.username = None
        self.api_token = None
        self.project_key = None
        self.is_enabled = False

    def configure(self, server: str, username: str, api_token: str, project_key: str) -> None:
        """Configure JIRA connection parameters."""
        print(f"[DEBUG] Configuring JIRA with:")
        print(f"[DEBUG] Server: {server}")
        print(f"[DEBUG] Username: {username}")
        print(f"[DEBUG] Project Key: {project_key}")
        
        self.server = server
        self.username = username
        self.api_token = api_token
        self.project_key = project_key
        self.is_enabled = True
        print("[DEBUG] JIRA configuration completed")

    def is_configured(self) -> bool:
        """Check if all required configuration is present."""
        return all([
            self.server,
            self.username,
            self.api_token,
            self.project_key,
            self.is_enabled
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

def search_jira_issues(query: str) -> Optional[str]:
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
            
        # Extract JIRA issue keys from the query
        issue_keys = extract_jira_keys(query, jira_config.project_key)
        print(f"[DEBUG] Extracted issue keys: {issue_keys}")
        
        # If no issue keys found, try text search
        if not issue_keys:
            # Remove common JIRA-related phrases to clean the query
            clean_query = clean_natural_language(query)
            jql = f'project = {jira_config.project_key} AND text ~ "{clean_query}"'
            print(f"[DEBUG] No issue keys found, performing text search: {clean_query}")
        else:
            # Search by issue keys
            keys_clause = ' OR '.join(f'key = "{key}"' for key in issue_keys)
            jql = f'({keys_clause})'
            print(f"[DEBUG] Searching by issue keys: {issue_keys}")
            
        print(f"[DEBUG] JIRA JQL: {jql}")
        issues = jira.search_issues(jql, maxResults=5)
        print(f"[DEBUG] JIRA Found Issues: {len(issues)}")
        
        if not issues:
            print("[DEBUG] No issues found")
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
        
        if len(issues) > 1:
            response = f"Found {len(issues)} JIRA issues:\n\n" + "\n\n".join(results)
        else:
            response = "\n".join(results)
            
        print(f"[DEBUG] JIRA search response:\n{response}")
        return response
        
    except Exception as e:
        print(f"Error searching JIRA: {str(e)}")
        print("[DEBUG] Full traceback:")
        traceback.print_exc()
        return None

# Initialize global JIRA config
jira_config = JiraConfig()
