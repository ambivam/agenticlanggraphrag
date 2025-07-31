from typing import Optional, List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from dotenv import load_dotenv
import os
import re

load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required for test case generation")

def create_test_case_prompt(issue: Dict[str, str]) -> str:
    """Create a detailed prompt for test case generation."""
    return f"""Feature: {issue['title']} ({issue['key']})

Background:
  Given the following JIRA issue:
    | Key         | {issue['key']} |
    | Title       | {issue['title']} |
    | Status      | {issue['status']} |
    | Description | {issue['description']} |

Please generate comprehensive test scenarios for this feature including:

1. Happy Path Cases:
   - Main functionality tests
   - Expected workflow scenarios

2. Positive Cases:
   - Valid input variations
   - Different user roles/permissions
   - Alternative successful paths

3. Edge Cases:
   - Boundary conditions
   - Maximum/minimum values
   - Resource limits
   - Timeout scenarios

4. Negative Cases:
   - Invalid inputs
   - Error handling
   - Missing required fields
   - Unauthorized access

5. Regression Cases:
   - Integration with existing features
   - Impact on related functionality
   - Data consistency checks

6. System Cases:
   - End-to-end workflows
   - Performance scenarios
   - Load testing scenarios

7. Unit Tests:
   - Component-level validation
   - Function-specific tests
   - Input/output validation

8. User Acceptance Tests:
   - Business requirement validation
   - User workflow scenarios
   - UI/UX validation

Format each scenario in Cucumber syntax with clear Given/When/Then steps and specific examples where relevant."""

def parse_jira_issues(content: str) -> List[Dict[str, str]]:
    """Parse JIRA content into structured issue data."""
    issues = []
    
    # Split content into individual issues
    issue_blocks = re.split(r'###\s+', content)
    
    for block in issue_blocks:
        if not block.strip():
            continue
            
        # Extract issue details using regex
        title_match = re.match(r'([^:]+):\s*(.+?)(?=\n|$)', block)
        status_match = re.search(r'\*\*Status:\*\*\s*([^|]+)', block)
        desc_match = re.search(r'\*\*Description:\*\*\s*(.+?)(?=\n\n|$)', block, re.DOTALL)
        
        if title_match:
            # Extract key and title separately
            key_title = title_match.group(0)
            key = key_title.split(':', 1)[0].strip()
            title = key_title.split(':', 1)[1].strip() if ':' in key_title else key_title
            
            issue = {
                'key': key,
                'title': title,
                'status': status_match.group(1).strip() if status_match else 'Unknown',
                'description': desc_match.group(1).strip() if desc_match else 'No description'
            }
            issues.append(issue)
            print(f"[DEBUG] Parsed JIRA issue: {issue}")
    
    return issues

def generate_test_cases(jira_content: str) -> Optional[str]:
    """Generate comprehensive BDD test cases for JIRA issues."""
    if not jira_content:
        return None
        
    # Extract JIRA issues from the content
    issues = parse_jira_issues(jira_content)
    if not issues:
        return None
        
    # Create test case generator agent
    try:
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0.7,
            model="gpt-4"
        )
    except Exception as e:
        print(f"Error initializing OpenAI: {str(e)}")
        return None
    
    test_cases = []
    for issue in issues:
        # Generate test cases for each issue
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior QA engineer tasked with creating comprehensive BDD test cases."),
            ("user", create_test_case_prompt(issue))
        ])
        
        try:
            # Get test cases from LLM
            messages = prompt.format_messages()
            response = llm.invoke(messages)
            test_cases.append(f"# Test Cases for {issue['key']}: {issue['title']}\n\n{response.content}\n\n")
        except Exception as e:
            print(f"Error generating test cases for {issue['key']}: {str(e)}")
            continue
    
    if not test_cases:
        return None
        
    return "\n".join(test_cases)
