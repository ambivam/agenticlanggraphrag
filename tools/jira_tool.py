from typing import Optional, Dict
from jira import JIRA
from langchain.tools import BaseTool

def get_jira_tool(server: str, username: str, api_token: str) -> Optional[BaseTool]:
    """Initialize and return a JIRA tool instance"""
    try:
        return JiraTool(server=server, username=username, api_token=api_token)
    except Exception as e:
        print(f"Error initializing JIRA tool: {str(e)}")
        return None

class JiraTool(BaseTool):
    name: str = "jira_search"
    description: str = "Search JIRA issues and get relevant information"
    jira: JIRA = None  # Type annotation for jira field
    
    def __init__(self, server: str, username: str, api_token: str):
        super().__init__()
        # Initialize the JIRA client
        try:
            self.jira = JIRA(
                server=server,
                basic_auth=(username, api_token)
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize JIRA client: {str(e)}")
    
    def _run(self, query: str) -> str:
        try:
            # Build a more comprehensive JQL query
            search_terms = query.replace('"', '').split()
            jql_conditions = []
            
            # Search in summary, description, and comments
            for term in search_terms:
                term_conditions = [
                    f'summary ~ "{term}"',
                    f'description ~ "{term}"',
                    f'comment ~ "{term}"'
                ]
                jql_conditions.append(f'({" OR ".join(term_conditions)})')
            
            jql = f'{" AND ".join(jql_conditions)} ORDER BY updated DESC'
            print(f"Executing JQL: {jql}")  # Debug info
            
            issues = self.jira.search_issues(jql, maxResults=5)
            
            if not issues:
                return "No relevant JIRA issues found. Try using different search terms or check your JIRA access permissions."
            
            # Format the results
            results = []
            for issue in issues:
                # Get issue details
                summary = issue.fields.summary
                status = str(issue.fields.status)
                description = issue.fields.description or 'No description'
                if len(description) > 200:  # Truncate long descriptions
                    description = description[:200] + '...'
                
                # Format the issue details
                results.append(
                    f"**{issue.key}**: {summary}\n"
                    f"- Status: {status}\n"
                    f"- Description: {description}\n"
                    f"- Created: {issue.fields.created[:10]}\n"
                    f"- Updated: {issue.fields.updated[:10]}\n"
                    f"- Link: {self.jira.server_url}/browse/{issue.key}\n"
                )
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error searching JIRA: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        # Async implementation would go here
        raise NotImplementedError("Async not implemented")
