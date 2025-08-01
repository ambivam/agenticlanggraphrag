from typing import Optional, List, Dict
from jira import JIRA
from datetime import datetime

class JIRAModule:
    def __init__(self):
        self.client = None
        
    def initialize(self, server: str, email: str, api_token: str) -> bool:
        try:
            self.client = JIRA(
                server=server,
                basic_auth=(email, api_token)
            )
            return True
        except Exception as e:
            print(f"Error connecting to JIRA: {str(e)}")
            return False
            
    def search_issues(self, jql_query: str, max_results: int = 10) -> Optional[str]:
        try:
            if not self.client:
                return "Error: JIRA client not initialized"
                
            issues = self.client.search_issues(jql_query, maxResults=max_results)
            
            if not issues:
                return "No issues found"
                
            # Format results
            results = []
            for issue in issues:
                issue_details = [
                    f"Key: {issue.key}",
                    f"Summary: {issue.fields.summary}",
                    f"Status: {issue.fields.status.name}",
                    f"Created: {issue.fields.created}",
                    f"Updated: {issue.fields.updated}",
                    f"Priority: {issue.fields.priority.name if issue.fields.priority else 'None'}",
                    f"Description: {issue.fields.description or 'No description'}"
                ]
                results.append("\n".join(issue_details))
                
            return "\n\n" + "-"*40 + "\n\n".join(results)
            
        except Exception as e:
            return f"Error searching JIRA issues: {str(e)}"
            
    def create_issue(self, project_key: str, summary: str, description: str, 
                    issue_type: str = "Task") -> Optional[str]:
        try:
            if not self.client:
                return "Error: JIRA client not initialized"
                
            issue_dict = {
                'project': {'key': project_key},
                'summary': summary,
                'description': description,
                'issuetype': {'name': issue_type},
            }
            
            new_issue = self.client.create_issue(fields=issue_dict)
            return f"Created issue: {new_issue.key}"
            
        except Exception as e:
            return f"Error creating JIRA issue: {str(e)}"
