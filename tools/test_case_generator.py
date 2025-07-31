from typing import Optional, List, Dict
import os
import re
import sys
import traceback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

# Initialize conversation memory
test_case_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required for test case generation")

def create_test_case_prompt(issue: Dict[str, str], scenarios_per_category: int = 10) -> str:
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
7. Focus on the search functionality requirements:
   - Search by keywords with AND/OR operators
   - Search in skills and work experience
   - Search with quotes for exact phrases
   - Advanced filtering options
   - Candidate status and availability
   - Location and language requirements
   - Salary expectations and current salary
   - Skills and experience matching

IMPORTANT: You MUST generate EXACTLY {scenarios_per_category} test scenarios for EACH category below. No more, no less.

Please generate a COMPREHENSIVE and DETAILED set of test scenarios focused on the advanced search functionality. The scenarios should cover:

1. Keyword Search Features:
   - Single keyword searches
   - Multiple keyword combinations
   - AND operator functionality
   - OR operator functionality
   - Exact phrase matching with quotes
   - Case sensitivity handling
   - Special character handling
   - Search result relevance

2. Search Scope:
   - Skills search
   - Work experience search
   - Education search
   - Combined field search
   - Profile-wide search
   - Field-specific search

3. Advanced Filters:
   - Country filter
   - City filter
   - Language and proficiency
   - Skills with suggestions
   - Salary expectations
   - Current salary
   - Employment status
   - Sector experience
   - Payment schemes
   - Relocation availability
   - Candidate status
   - Recruitment source
   - Recruiter assignment

4. Result Handling:
   - Result ordering
   - Result filtering
   - Result pagination
   - Result accuracy
   - Performance with large datasets
   - Response time requirements

REQUIREMENTS FOR EACH SCENARIO:
1. Must be specific and actionable
2. Must have clear success criteria
3. Must include realistic test data
4. Must cover edge cases where applicable
5. Must validate business rules

REMEMBER: Generate EXACTLY {scenarios_per_category} scenarios for each category.

Test Categories:

# Happy Path Cases (Core Functionality)
- Main success scenarios from description
- Primary user flows mentioned
- Expected behavior for each requirement
- Different user roles' success paths
- Specific workflows described

# Positive Cases (Valid Variations)
- Alternative valid inputs described
- Different valid user flows
- Optional feature combinations
- Valid data variations mentioned
- Permitted user role variations

# Edge Cases (Boundary Testing)
- Minimum/maximum values from requirements
- Resource limits mentioned
- Timing conditions in description
- Concurrent operations
- Integration edge cases
- Load conditions specified

# Negative Cases (Error Handling)
- Invalid inputs for each field
- Missing required data
- System errors mentioned
- Network issues
- Security violations
- Error scenarios described

# Regression Cases (Impact Analysis)
- Integration points from description
- Dependent features mentioned
- Data consistency requirements
- State transitions
- Backward compatibility needs
- Migration scenarios described

# System Cases (End-to-End)
- Performance requirements mentioned
- Security constraints described
- Data persistence rules
- State management scenarios
- Integration flows
- Monitoring requirements

# Unit Tests (Component Level)
- Function inputs/outputs from description
- State transitions mentioned
- Business logic rules
- Validation requirements
- Component interactions
- Error handling specifics

# User Acceptance Tests (Business Validation)
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

def format_test_cases(issue: Dict[str, str], content: str, scenarios_per_category: int) -> str:
    """Format test cases with HTML and Gherkin syntax highlighting.
    
    Args:
        issue: Dictionary containing JIRA issue details
        content: Raw test case content from LLM
        scenarios_per_category: Number of scenarios per category
        
    Returns:
        str: Formatted HTML content with test cases
    """
    # Add JIRA issue details section first
    formatted_content = [f"""
<details class='issue-details' open>
<summary><strong>📋 JIRA Issue Details</strong></summary>

| Field | Value |
|-------|-------|
| Key | {issue.get('key', 'N/A')} |
| Title | {issue.get('title', 'N/A')} |
| Status | {issue.get('status', 'N/A')} |

**Description:**
```
{issue.get('description', 'No description provided')}
```
</details>

## 🎯 Generated Test Cases
"""]
    
    # Process test case sections
    sections = {}
    
    # Split content into sections using regex
    section_pattern = re.compile(r'^# [^\n]+$', re.MULTILINE)
    section_matches = list(section_pattern.finditer(content))
    
    # Debug: Print raw content
    print("\n=== Raw Content ===\n")
    print(content)
    print("\n=== End Raw Content ===\n")
    
    # Process each section
    for i in range(len(section_matches)):
        # Get section title
        section_start = section_matches[i].start()
        section_title = content[section_start:section_matches[i].end()].strip()
        
        # Get section content
        if i < len(section_matches) - 1:
            section_end = section_matches[i + 1].start()
        else:
            section_end = len(content)
        
        # Extract and clean section content
        section_content = content[section_matches[i].end():section_end].strip()
        sections[section_title] = section_content
        
        # Debug: Print section info
        print(f"\n=== Section Found ===\nTitle: {section_title}\nContent:\n{section_content}\n====================\n")
    
    # If no sections found, treat entire content as Happy Path
    if not sections and content.strip():
        sections['# Happy Path Cases'] = content.strip()
    
    # Process each test case section
    test_case_sections = [
        "# Happy Path Cases",
        "# Positive Cases", 
        "# Edge Cases",
        "# Negative Cases",
        "# Regression Cases",
        "# System Cases",
        "# Unit Tests",
        "# User Acceptance Tests"
    ]
    
    for title in test_case_sections:
        section_content = sections.get(title, '')
        if not section_content.strip():
            section_content = f"Scenario: Basic test for {title.replace('#', '').strip()}\nGiven the system is ready\nWhen testing the {title.replace('#', '').strip().lower()}\nThen verify the expected behavior"
        
        # Clean up section content
        clean_section = section_content.strip()
        
        # Remove gherkin code block markers if present
        if clean_section.startswith('```gherkin'):
            clean_section = clean_section[len('```gherkin'):]
        if clean_section.endswith('```'):
            clean_section = clean_section[:-3]
            
        # Remove any leading/trailing whitespace after code block removal
        clean_section = clean_section.strip()
        
        # Ensure proper newline handling for tables and indentation
        lines = []
        for line in clean_section.split('\n'):
            # Preserve table formatting and indentation
            if '|' in line or line.startswith('    '):
                lines.append(line)  # Keep original spacing
            else:
                lines.append(line.strip())
        clean_section = '\n'.join(lines)
        
        # Add formatted section
        emoji_map = {
            '# Happy Path Cases': '🎯',
            '# Positive Cases': '✅',
            '# Edge Cases': '🔄',
            '# Negative Cases': '❌',
            '# Regression Cases': '🔄',
            '# System Cases': '🌐',
            '# Unit Tests': '🧪',
            '# User Acceptance Tests': '👥'
        }
        
        emoji = emoji_map.get(title, 'ℹ️')
        scenario_count = clean_section.count('Scenario:')
        
        formatted_content.append(f"""
<details class='scenario-section'>
<summary><strong>{emoji} {title.replace('#', '').strip()}</strong> ({scenario_count} scenarios)</summary>

```gherkin
{clean_section}
```
</details>""")
    
    return '\n'.join(formatted_content)

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
        status_match = re.search(r'\*\*Status:\*\*\s*([^|\n]+)', block)
        desc_match = re.search(r'\*\*Description:\*\*\s*([\s\S]+?)(?=\n\*\*|$)', block)
        
        # Clean up description if found
        description = ''
        if desc_match:
            description = desc_match.group(1).strip()
            # Remove JIRA color formatting
            description = re.sub(r'\{color:[^}]+\}([^{]+)\{color\}', r'\1', description)
            # Remove image tags
            description = re.sub(r'!.*?!', '', description)
            # Clean up extra whitespace
            description = re.sub(r'\n{3,}', '\n\n', description)
            description = description.strip()
        
        if title_match:
            # Extract key and title separately
            key_title = title_match.group(0)
            key = key_title.split(':', 1)[0].strip()
            title = key_title.split(':', 1)[1].strip() if ':' in key_title else key_title
            
            issue = {
                'key': key,
                'title': title,
                'status': status_match.group(1).strip() if status_match else 'Unknown',
                'description': description if description else 'No description provided'
            }
            issues.append(issue)
            print(f"[DEBUG] Parsed JIRA issue: {issue}")
    
    return issues

def generate_test_cases(jira_content: str, temperature: float = 0.7, scenarios_per_category: int = 10, continue_previous: bool = False) -> Optional[str]:
    """Generate BDD test cases from JIRA content using GPT-4.
    
    Args:
        jira_content: String containing JIRA issue details
        temperature: Float between 0.0 and 1.0 controlling output creativity (default: 0.7)
        scenarios_per_category: Number of scenarios to generate per category (default: 10)
        continue_previous: Whether to use previous test cases as context (default: False)
        
    Returns:
        Optional[str]: HTML-formatted test cases with Gherkin syntax highlighting,
                      or None if generation fails
    """
    if not jira_content:
        return None
        
    try:
        # Get chat history if continuing from previous generation
        chat_history = ""
        if continue_previous:
            messages = test_case_memory.chat_memory.messages
            if messages:
                chat_history = "\nPrevious test cases generated:\n" + "".join([msg.content for msg in messages])
            
        # Extract JIRA issues from the content
        issues = parse_jira_issues(jira_content)
        if not issues:
            return "<details class='error-section' open>\n<summary><strong>⚠️ No JIRA Issues Found</strong></summary>\nNo valid JIRA issues were found in the provided content.</details>"

        # Create test case generator agent
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            model="gpt-4",
            max_tokens=4000  # Limit output size
        )
        
        test_cases = []
        for issue in issues:
            # Generate test cases for each issue
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a senior QA engineer tasked with creating comprehensive BDD test cases."),
                ("user", create_test_case_prompt(issue, scenarios_per_category))
            ])
            
            # Get test cases from LLM
            messages = prompt.format_messages(scenarios_per_category=scenarios_per_category)
            response = llm.invoke(messages)
            
            # Debug: Print raw response
            print("\n=== Raw LLM Response ===\n")
            print(response.content)
            print("\n=== End Raw Response ===\n")
            
            # Process and format the test cases
            content = response.content
            if len(content) > 32000:  # If content is too large
                print(f"[WARNING] Large output detected for {issue['key']}, truncating...")
                content = content[:32000] + "\n\n[Output truncated due to length]\n"
            
            # Store the generated test cases in memory for future reference
            test_case_memory.save_context(
                {"input": jira_content},
                {"output": response.content}
            )
            
            # Format the test cases with HTML and Gherkin highlighting
            formatted_content = format_test_cases(issue, content, scenarios_per_category)
            test_cases.append(formatted_content)
            
        return "\n".join(test_cases) if test_cases else None
            
    except Exception as e:
        error_details = f"Error generating test cases: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_details, file=sys.stderr)
        return f"<details class='error-section' open>\n<summary><strong>❌ Error Generating Test Cases</strong></summary>\n\n```\n{error_details}\n```\n</details>"
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
                ("user", create_test_case_prompt(issue, scenarios_per_category))
            ])
            
            # Get test cases from LLM
            messages = prompt.format_messages(scenarios_per_category=scenarios_per_category)
            response = llm.invoke(messages)
            
            # Debug: Print raw response
            print("\n=== Raw LLM Response ===")
            print(response.content)
            print("=== End Raw Response ===")
            
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
                ("Happy Path Cases", None, None, scenarios_per_category),
                ("Positive Cases", None, None, scenarios_per_category),
                ("Edge Cases", None, None, scenarios_per_category),
                ("Negative Cases", None, None, scenarios_per_category),
                ("Regression Cases", None, None, scenarios_per_category),
                ("System Cases", None, None, scenarios_per_category),
                ("Unit Tests", None, None, scenarios_per_category),
                ("User Acceptance Tests", None, None, scenarios_per_category)
            ]
            # Process each category
            for title, _, _, min_scenarios in categories:
                section = ""
                scenario_count = 0  # Initialize counter
                try:
                    # Find section using regex with flexible matching
                    patterns = [
                        # Try exact match
                        f'^{re.escape(title)}\s*(?:\(.*?\))?[:\s]*$',
                        # Try with leading #
                        f'^#\s*{re.escape(title)}\s*(?:\(.*?\))?[:\s]*$',
                        # Try with emoji prefix
                        f'^[\u2600-\U0001F9FF]\s*{re.escape(title)}\s*(?:\(.*?\))?[:\s]*$',
                        # Try case insensitive
                        f'^(?i){re.escape(title)}\s*(?:\(.*?\))?[:\s]*$'
                    ]
                    
                    section = None
                    for pattern in patterns:
                        # Split content into sections by any markdown header or double newline
                        sections = re.split(r'\n(?:#{1,6}\s|\n\s*\n)', content)
                        
                        # Find section that starts with our pattern
                        for s in sections:
                            if re.search(pattern, s.strip().split('\n')[0], re.MULTILINE):
                                section = '\n'.join(s.strip().split('\n')[1:])
                                break
                        
                        if section:
                            break
                    
                    if not section:
                        raise ValueError(f"Could not find section '{title}' using any pattern")
                    
                    # Verify section has content
                    if not section.strip():
                        raise ValueError("Empty section content")
                    
                    # Count scenarios and ensure minimum
                    scenario_count = section.count("Scenario:")
                    scenarios_needed = max(0, min_scenarios - scenario_count)
                    
                    if scenarios_needed > 0:
                        print(f"[WARNING] {title} has only {scenario_count} scenarios (minimum {min_scenarios})")
                        # Add placeholder scenarios if needed
                        placeholder_scenarios = []
                        for i in range(scenarios_needed):
                            placeholder_scenarios.append(
                                f"Scenario: Additional test for {title} #{i+1}\n"
                                f"Given the search functionality requirements\n"
                                f"When implementing missing test cases\n"
                                f"Then add specific scenarios for {title.lower()}\n"
                            )
                        section = "\n\n".join([section] + placeholder_scenarios)
                        scenario_count = min_scenarios  # Update count
                    
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
                # Add emoji based on category
                emoji_map = {
                    'Happy Path Cases': '🎯',
                    'Positive Cases': '✅',
                    'Edge Cases': '🔄',
                    'Negative Cases': '❌',
                    'Regression Cases': '🔄',
                    'System Cases': '🌐',
                    'Unit Tests': '🧪',
                    'User Acceptance Tests': '👥'
                }
                emoji = emoji_map.get(title, 'ℹ️')
                
                formatted_content.append(f"""
<details class='scenario-section'>
<summary><strong>{emoji} {title}</strong> ({scenario_count} scenarios)</summary>

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
