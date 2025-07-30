from typing import Optional
from jira import JIRA
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

class JiraConfig:
    def __init__(self):
        self.server = None
        self.username = None
        self.api_token = None
        self.project_key = None
        self.is_enabled = False

    def configure(self, server: str, username: str, api_token: str, project_key: str):
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
        return all([
            self.server,
            self.username,
            self.api_token,
            self.project_key,
            self.is_enabled
        ])

jira_config = JiraConfig()

def get_jira_tool():
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

def search_jira_issues(query: str) -> Optional[str]:
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
            
        # Search for issues in the configured project
        # Parse multiple issue keys if present
        issue_keys = [key.strip() for key in query.split(',')]
        
        # Check if all items look like JIRA keys
        all_keys_valid = all(key.strip().startswith(jira_config.project_key + '-') for key in issue_keys)
        
        if all_keys_valid:
            # If all items are issue keys, search by multiple keys
            keys_clause = ' OR '.join(f'key = "{key.strip()}"' for key in issue_keys)
            jql = f'({keys_clause})'
            print(f"[DEBUG] Searching by issue keys: {issue_keys}")
        else:
            # Otherwise do a text search
            jql = f'project = {jira_config.project_key} AND text ~ "{query}"'
            print(f"[DEBUG] Performing text search: {query}")
            
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
