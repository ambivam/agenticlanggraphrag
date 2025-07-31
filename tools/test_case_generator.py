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

Please generate a COMPREHENSIVE and DETAILED set of test scenarios for this feature. For each category below, provide AT LEAST 8 distinct test scenarios. Each scenario must be specific to the feature described in the JIRA issue and not generic.

Analyze the issue description thoroughly and consider all possible user interactions, system behaviors, and potential edge cases. Format all scenarios in Gherkin/Cucumber syntax with clear Given/When/Then steps.

1. 🎯 Happy Path Cases (Core Functionality):
   - Primary user workflows
   - Main feature interactions
   - Expected successful scenarios
   - Standard use cases
   - Basic CRUD operations if applicable
   - Common user journeys
   - Typical data scenarios
   - Expected state transitions

2. ✅ Positive Cases (Valid Variations):
   - Different valid input combinations
   - Various user roles and permissions
   - Alternative successful paths
   - Different data types and formats
   - Multiple language/locale scenarios
   - Valid configuration variations
   - Cross-browser/platform scenarios
   - Integration with other features

3. 🔄 Edge Cases (Boundary Testing):
   - Minimum/maximum values
   - Resource limits (memory, CPU, storage)
   - Timeout conditions
   - Concurrent user access
   - Large data sets
   - Network conditions (slow, intermittent)
   - Browser cache/cookie scenarios
   - State transition boundaries

4. ❌ Negative Cases (Error Handling):
   - Invalid inputs and formats
   - Missing required fields
   - Unauthorized access attempts
   - Invalid state transitions
   - System errors and failures
   - Network errors
   - Database errors
   - API errors

5. 🔄 Regression Cases (Impact Analysis):
   - Integration with existing features
   - Data consistency checks
   - Backward compatibility
   - Configuration changes
   - Database schema updates
   - API version compatibility
   - UI/UX consistency
   - Performance baseline

6. 🌐 System Cases (End-to-End):
   - Complete user journeys
   - Performance under load
   - Scalability scenarios
   - Data backup/recovery
   - System upgrades
   - Security scenarios
   - Integration points
   - Monitoring and logging

7. 🧪 Unit Tests (Component Level):
   - Function parameter validation
   - Return value verification
   - State management
   - Event handling
   - Error conditions
   - Component initialization
   - Resource cleanup
   - Module interactions

8. 👥 User Acceptance Tests (Business Validation):
   - Business rule compliance
   - Workflow validation
   - UI/UX requirements
   - Accessibility compliance
   - Data privacy requirements
   - Regulatory compliance
   - Reporting accuracy
   - User experience goals

Format each scenario following this template:

  Scenario: [Clear descriptive title]
    Given [precise context and prerequisites]
    When [specific user or system actions]
    Then [expected outcomes in detail]
    And [additional verification steps if needed]

Provide specific examples and test data where relevant. Include boundary values, equivalence classes, and realistic test data that matches the business context."""

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
            
            # Split response into categories
            categories = [
                ("🎯 Happy Path Cases", "1.", "2."),
                ("✅ Positive Cases", "2.", "3."),
                ("🔄 Edge Cases", "3.", "4."),
                ("❌ Negative Cases", "4.", "5."),
                ("🔄 Regression Cases", "5.", "6."),
                ("🌐 System Cases", "6.", "7."),
                ("🧪 Unit Tests", "7.", "8."),
                ("👥 User Acceptance Tests", "8.", None)
            ]
            # Format test cases with collapsible sections
            formatted_content = [f"## 🎯 Test Cases for {issue['key']}: {issue['title']}\n"]
            
            content = response.content
            try:
                # Split response into categories
                categories = [
                    ("🎯 Happy Path Cases", "1.", "2."),
                    ("✅ Positive Cases", "2.", "3."),
                    ("🔄 Edge Cases", "3.", "4."),
                    ("❌ Negative Cases", "4.", "5."),
                    ("🔄 Regression Cases", "5.", "6."),
                    ("🌐 System Cases", "6.", "7."),
                    ("🧪 Unit Tests", "7.", "8."),
                    ("👥 User Acceptance Tests", "8.", None)
                ]
                
                for title, start, end in categories:
                    section = content.split(start)[1].split(end)[0] if end else content.split(start)[1]
                    formatted_content.append(f"""
<details class='scenario-section'>
<summary><strong>{title}</strong></summary>

```gherkin
{start}{section.strip()}
```

</details>""")
                
                test_cases.append("\n".join(formatted_content))
            except Exception as e:
                print(f"Error generating test cases: {str(e)}")
                print("Full traceback:")
                traceback.print_exc()
        except Exception as e:
            print(f"Error generating test cases for {issue['key']}: {str(e)}")
            continue
    
    if not test_cases:
        return None
        
    return "\n".join(test_cases)
