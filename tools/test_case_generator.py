from typing import Optional, List, Dict
import os
import re
import sys
import traceback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from dotenv import load_dotenv
import re

load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required for test case generation")

def create_test_case_prompt(issue: Dict[str, str]) -> str:
    """Create a detailed prompt for test case generation.
    
    Args:
        issue: Dictionary containing JIRA issue details (key, title, status, description)
        
    Returns:
        str: Formatted prompt for test case generation
    """
    return f"""Generate test cases for the following JIRA issue:

Issue Details:
    | Key         | {issue['key']} |
    | Title       | {issue['title']} |
    | Status      | {issue['status']} |
    | Description | {issue['description']} |

IMPORTANT INSTRUCTIONS:
1. Carefully analyze BOTH the title AND description to understand the complete feature scope
2. Extract all requirements, constraints, and business rules from the description
3. Consider any technical details or implementation notes mentioned
4. Identify dependencies and integration points described
5. Pay attention to any specific user roles or permissions mentioned
6. Note any performance, security, or compliance requirements

Please generate a COMPREHENSIVE and DETAILED set of test scenarios. For each category below, provide AT LEAST 20 distinct test scenarios that are:
- Specific to the feature described in BOTH title and description
- Cover all aspects mentioned in the description
- Address both functional and non-functional requirements
- Consider all possible user roles and permissions
- Test all mentioned integrations and dependencies

Categories to cover:

1. 🎯 Happy Path Cases (Core Functionality)
   - Main success scenarios from description
   - Primary user flows mentioned
   - Expected behavior for each requirement
   - Different user roles' success paths
   - Specific workflows described

2. ✅ Positive Cases (Valid Variations)
   - Alternative valid inputs described
   - Different valid user flows
   - Optional feature combinations
   - Valid data variations mentioned
   - Permitted user role variations

3. 🔄 Edge Cases (Boundary Testing)
   - Minimum/maximum values from requirements
   - Resource limits mentioned
   - Timing conditions in description
   - Concurrent operations
   - Integration edge cases
   - Load conditions specified

4. ❌ Negative Cases (Error Handling)
   - Invalid inputs for each field
   - Missing required data
   - System errors mentioned
   - Network issues
   - Security violations
   - Error scenarios described

5. 🔄 Regression Cases (Impact Analysis)
   - Integration points from description
   - Dependent features mentioned
   - Data consistency requirements
   - State transitions
   - Backward compatibility needs
   - Migration scenarios described

6. 🌐 System Cases (End-to-End)
   - Performance requirements mentioned
   - Security constraints described
   - Data persistence rules
   - State management scenarios
   - Integration flows
   - Monitoring requirements

7. 🧪 Unit Tests (Component Level)
   - Function inputs/outputs from description
   - State transitions mentioned
   - Business logic rules
   - Validation requirements
   - Component interactions
   - Error handling specifics

8. 👥 User Acceptance Tests (Business Validation)
   - Business requirements from description
   - User workflows mentioned
   - Domain rules specified
   - Compliance requirements
   - User experience criteria
   - Business process validations

Use this Gherkin template and ensure each scenario references specific requirements from the description:
```gherkin
Scenario: [Clear title referencing specific requirement]
Given [preconditions from description]
When [actions based on described workflows]
Then [outcomes matching requirements]
And [validations for specific criteria]
```

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

def generate_test_cases(jira_content: str, temperature: float = 0.7) -> Optional[str]:
    """Generate BDD test cases from JIRA content using GPT-4.
    
    Args:
        jira_content: String containing JIRA issue details
        temperature: Float between 0.0 and 1.0 controlling output creativity (default: 0.7)
        
    Returns:
        Optional[str]: HTML-formatted test cases with Gherkin syntax highlighting,
                      or None if generation fails
        Formatted test cases or None if generation fails
    """
    if not jira_content:
        return None
        
    # Extract JIRA issues from the content
    issues = parse_jira_issues(jira_content)
    if not issues:
        return None
        
    # Parse JIRA content first
    try:
        issues = parse_jira_issues(jira_content)
        if not issues:
            return "<details class='error-section' open>\n<summary><strong>⚠️ No JIRA Issues Found</strong></summary>\nNo valid JIRA issues were found in the provided content.</details>"
    except Exception as e:
        error_details = f"Error parsing JIRA content: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_details, file=sys.stderr)
        return f"<details class='error-section' open>\n<summary><strong>❌ Error Parsing JIRA Content</strong></summary>\n\n```\n{error_details}\n```\n</details>"

    # Create test case generator agent
    try:
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            model="gpt-4",
            max_tokens=4000  # Limit output size
        )
    except Exception as e:
        error_details = f"Error initializing OpenAI: {str(e)}"
        print(error_details, file=sys.stderr)
        return f"<details class='error-section' open>\n<summary><strong>❌ OpenAI Error</strong></summary>\n\n```\n{error_details}\n```\n</details>"
    
    test_cases = []
    for issue in issues:
        try:
            # Generate test cases for each issue
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a senior QA engineer tasked with creating comprehensive BDD test cases."),
                ("user", create_test_case_prompt(issue))
            ])
            
            # Get test cases from LLM
            messages = prompt.format_messages()
            response = llm.invoke(messages)
            
            # Split response into categories and handle large outputs
            content = response.content
            if len(content) > 32000:  # If content is too large
                print(f"[WARNING] Large output detected for {issue['key']}, truncating...")
                content = content[:32000] + "\n\n[Output truncated due to length]\n"
                
            # Add JIRA issue details section
            formatted_content = ["""
<details class='issue-details' open>
<summary><strong>📋 JIRA Issue Details</strong></summary>

| Field | Value |
|-------|-------|
| Key | {key} |
| Title | {title} |
| Status | {status} |

**Description:**
```
{description}
```
</details>

## 🎯 Generated Test Cases
""".format(
                key=issue.get('key', 'N/A'),
                title=issue.get('title', 'N/A'),
                status=issue.get('status', 'N/A'),
                description=issue.get('description', 'No description provided')
            )]
            
            # Define categories with minimum scenario requirements
            categories = [
                ("🎯 Happy Path Cases", "1.", "2.", 5),  # Min scenarios per category
                ("✅ Positive Cases", "2.", "3.", 5),
                ("🔄 Edge Cases", "3.", "4.", 3),
                ("❌ Negative Cases", "4.", "5.", 3),
                ("🔄 Regression Cases", "5.", "6.", 2),
                ("🌐 System Cases", "6.", "7.", 2),
                ("🧪 Unit Tests", "7.", "8.", 2),
                ("👥 User Acceptance Tests", "8.", None, 2)
            ]
            # Process each category
            for title, start, end, min_scenarios in categories:
                section = ""
                try:
                    # Extract section safely
                    parts = content.split(start)
                    if len(parts) < 2:
                        raise ValueError(f"Could not find section start marker '{start}'")
                        
                    section = parts[1]  # Take the part after the start marker
                    
                    if end:
                        end_parts = section.split(end)
                        if len(end_parts) < 1:
                            raise ValueError(f"Could not find section end marker '{end}'")
                        section = end_parts[0]  # Take the part before the end marker
                    
                    # Verify section has content
                    if not section.strip():
                        raise ValueError("Empty section content")
                    
                    # Verify minimum scenarios
                    scenario_count = section.count("Scenario:")
                    if scenario_count < min_scenarios:
                        print(f"[WARNING] {title} has only {scenario_count} scenarios (minimum {min_scenarios})")
                        # Add placeholder scenarios if needed
                        if scenario_count == 0:
                            section = f"Scenario: Placeholder for {title}\nGiven no scenarios were generated\nWhen reviewing the output\nThen add more specific scenarios\n\n{section}"
                    
                    # Truncate very large sections
                    if len(section) > 8000:
                        section = section[:8000] + "\n\n[Section truncated due to length]\n"
                    
                    section = section.strip()
                    
                except Exception as e:
                    error_msg = f"Error processing section: {str(e)}"
                    print(f"Error in {title}: {str(e)}")
                    section = f"Scenario: Error in {title}\nGiven an error occurred: {str(e)}\nWhen processing the section\nThen fix the issue and retry"
                
                # Clean up and format the section
                clean_section = section
                # Remove any existing gherkin tags
                clean_section = clean_section.replace('```gherkin', '').replace('```', '')
                # Remove category headers that might have been included
                clean_section = re.sub(r'^[\d\.]+\s+.*?\n', '', clean_section, flags=re.MULTILINE)
                # Remove any emojis that might have been included
                clean_section = re.sub(r'[\U0001F300-\U0001F9FF]', '', clean_section)
                # Fix extra newlines
                clean_section = re.sub(r'\n{3,}', '\n\n', clean_section)
                # Fix indentation
                clean_section = '\n'.join(line.strip() for line in clean_section.split('\n'))
                clean_section = clean_section.strip()
                
                # Add formatted section to output with proper syntax highlighting
                formatted_content.append(f"""
<details class='scenario-section'>
<summary><strong>{title}</strong> ({scenario_count} scenarios)</summary>

```gherkin
{clean_section}
```
</details>
""")
                
            test_cases.append("\n".join(formatted_content))
        except Exception as e:
            error_details = f"Error generating test cases: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(error_details, file=sys.stderr)
            return f"<details class='error-section' open>\n<summary><strong>❌ Error Generating Test Cases</strong></summary>\n\n```\n{error_details}\n```\n</details>"
    
    if not test_cases:
        return None
        
    return "\n".join(test_cases)
