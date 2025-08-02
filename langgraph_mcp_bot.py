import os
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, Optional, List, Dict, Any

from tools.rag_tool import get_rag_chain
from tools.mysql_tool import get_mysql_agent
from tools.serpapi_tool import get_serp_tool
from tools.jira_tool import get_jira_tools, jira_config
from module_manager import module_manager

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END, START

load_dotenv()

class ChatState(TypedDict, total=False):
    input: str
    rag_context: Optional[str]
    sql_context: Optional[str]
    serp_context: Optional[str]
    jira_context: Optional[str]
    test_cases: Optional[str]
    final_answer: Optional[str]
    found: bool

# Initialize tools
def get_tools(temperature=0.7):
    """Get the tools for the agent based on enabled modules."""
    tools = []
    
    print("\nGetting Tools:")
    print(f"RAG Enabled: {module_manager.is_enabled('rag')}")
    print(f"SQL Enabled: {module_manager.is_enabled('sql')}")
    print(f"Search Enabled: {module_manager.is_enabled('search')}")
    print(f"JIRA Enabled: {module_manager.is_enabled('jira')}")
    
    if module_manager.is_enabled('rag'):
        print("Adding RAG tool...")
        tools.append(get_rag_chain())
    if module_manager.is_enabled('sql'):
        print("Adding SQL tool...")
        tools.append(get_mysql_agent())
    if module_manager.is_enabled('search'):
        print("Adding Search tool...")
        tools.append(get_serp_tool())
    if module_manager.is_enabled('jira'):
        print("Adding JIRA tools...")
        jira_tools = get_jira_tools()
        print(f"JIRA tools found: {len(jira_tools)}")
        tools.extend(jira_tools)
    
    print(f"Total tools: {len(tools)}")
    return tools

tools = get_tools()

def invoke_tool(tool, input_text, **kwargs):
    """Invoke a tool that could be either a function or an object with invoke method.
    
    Args:
        tool: The tool to invoke
        input_text: The input text for the tool
        **kwargs: Additional arguments to pass to the tool
    """
    if callable(tool) and not hasattr(tool, 'invoke'):
        return tool(input_text, **kwargs)
    return tool.invoke(input_text, **kwargs)

def rag_node(state: ChatState) -> ChatState:
    """RAG node that processes knowledge base queries."""
    tools = get_tools()
    if not module_manager.is_enabled('rag') or not tools:
        return state
    
    # Initialize state fields if not present
    if "found" not in state:
        state["found"] = False
    if "rag_context" not in state:
        state["rag_context"] = None
    if "sql_context" not in state:
        state["sql_context"] = None
    if "serp_context" not in state:
        state["serp_context"] = None
    if "final_answer" not in state:
        state["final_answer"] = None
        
    # Find RAG tool
    rag_tool = None
    for tool in tools:
        if hasattr(tool, 'is_rag_tool'):
            rag_tool = tool
            break
            
    if not rag_tool:
        return state
        
    response = invoke_tool(rag_tool, state["input"])
    if isinstance(response, dict):
        response_text = response.get('result', '')
    else:
        response_text = str(response)
    
    if "I don't know" in response_text or len(response_text.strip()) < 5:
        state["rag_context"] = None
        state["found"] = False
    else:
        state["rag_context"] = response_text
        state["found"] = True
    return state

def mysql_node(state: ChatState) -> ChatState:
    """MySQL node that processes SQL queries."""
    tools = get_tools()
    if not module_manager.is_enabled('sql') or not tools:
        return state
    
    try:
        sql_tool = None
        for tool in tools:
            if hasattr(tool, 'is_sql_agent'):
                sql_tool = tool
                break
        
        if not sql_tool:
            return state
            
        response = invoke_tool(sql_tool, state["input"])
        result = str(response).strip()
        
        # Check if we got a meaningful result
        meaningful = (
            len(result) > 0 and
            "error" not in result.lower() and
            "i don't know" not in result.lower() and
            "no relevant information" not in result.lower() and
            "could not find" not in result.lower() and
            "invalid format" not in result.lower() and
            "could not parse" not in result.lower() and
            "no results" not in result.lower()
        )
            
        state["sql_context"] = result if meaningful else None
        state["found"] = meaningful
        
    except Exception as e:
        print(f"MySQL Error: {str(e)}")
        state["sql_context"] = None
        state["found"] = False
    return state

def serp_node(state: ChatState) -> ChatState:
    """Web search node that processes search queries."""
    tools = get_tools()
    if not module_manager.is_enabled('search') or not tools:
        return state
    
    search_tool = None
    for tool in tools:
        if hasattr(tool, 'is_search_tool'):
            search_tool = tool
            break
            
    if not search_tool:
        return state
        
    response = invoke_tool(search_tool, state["input"])
    if response:  # Only set context if we got a valid response
        state["serp_context"] = response
        state["found"] = True
    else:
        state["serp_context"] = None
        state["found"] = False
    return state

def jira_node(state: ChatState) -> ChatState:
    """JIRA node that processes JIRA queries."""
    print(f"JIRA Node - Input: {state.get('input')}")
    print(f"JIRA Node - Module Enabled: {module_manager.is_enabled('jira')}")
    
    tools = get_tools()
    print(f"JIRA Node - Tools Found: {len(tools)}")
    
    if not module_manager.is_enabled('jira') or not tools:
        print("JIRA Node - Module disabled or no tools found")
        return state
    
    print(f"JIRA Node - Config Status: {jira_config.is_configured()}")
    print(f"JIRA Node - Project Key: {jira_config.project_key}")
    
    if not jira_config.is_configured():
        print("JIRA Node - JIRA not configured")
        state["jira_context"] = None
        state["found"] = False
        return state
    
    try:
        # Find JIRA tool
        jira_tool = None
        test_case_gen = None
        for tool in tools:
            if hasattr(tool, 'is_jira_tool'):
                jira_tool = tool
                print("JIRA Node - Found JIRA tool")
            elif hasattr(tool, 'is_test_case_generator'):
                test_case_gen = tool
                print("JIRA Node - Found test case generator")
                
        if not jira_tool:
            print("JIRA Node - No JIRA tool found")
            return state
            
        # Search JIRA issues
        print("JIRA Node - Searching JIRA...")
        response = invoke_tool(jira_tool, state["input"])
        print(f"JIRA Node - Search Response: {bool(response)}")
        
        if response:  # Only set context if we got a valid response
            state["jira_context"] = response
            state["found"] = True
            print("JIRA Node - Found JIRA content")
            
            # Generate test cases if available
            if test_case_gen and state.get("jira_context"):
                print("JIRA Node - Generating test cases...")
                state["test_cases"] = invoke_tool(test_case_gen, state["jira_context"])
                print(f"JIRA Node - Test Cases Generated: {bool(state.get('test_cases'))}")
        else:
            state["jira_context"] = None
            state["found"] = False
            print("JIRA Node - No JIRA content found")
    except Exception as e:
        print(f"JIRA Error: {str(e)}\nTraceback:\n{traceback.format_exc()}")
        state["jira_context"] = None
        state["found"] = False
    return state

def test_case_node(state: ChatState) -> ChatState:
    """Generate test cases for JIRA issues."""
    tools = get_tools()
    if not module_manager.is_enabled('jira') or not tools:
        return state
        
    test_case_gen = None
    for tool in tools:
        if hasattr(tool, 'is_test_case_generator'):
            test_case_gen = tool
            break
            
    if not test_case_gen:
        return state
        
    # Only generate test cases if we have JIRA content
    if state.get("jira_context"):
        # Check if this is a continuation request
        continue_previous = False
        query = state.get("input", "").lower()
        if any(x in query for x in ["more", "additional", "continue", "generate more"]):
            continue_previous = True
        
        response = invoke_tool(
            test_case_gen,
            state["jira_context"],
            temperature=state.get("llm_temperature", 0.7),
            scenarios_per_category=state.get("scenarios_per_category", 10),
            continue_previous=continue_previous
        )
        
        if response:  # Only set context if we got a valid response
            state["test_cases"] = response
            state["found"] = True
        else:
            state["test_cases"] = None
            state["found"] = False
    return state

def final_answer_node(state: ChatState) -> ChatState:
    """Combine all available context into a final answer."""
    state['final_answer'] = ""
    sources = []
    
    if module_manager.is_enabled('rag') and state.get("rag_context"):
        sources.append(f"From knowledge base: {state['rag_context']}")
    
    if module_manager.is_enabled('sql') and state.get("sql_context"):
        sources.append(f"From database: {state['sql_context']}")
    
    if module_manager.is_enabled('search') and state.get("serp_context"):
        sources.append(f"From web search: {state['serp_context']}")
    
    if module_manager.is_enabled('jira'):
        if state.get("jira_context"):
            sources.append(f"From JIRA: {state['jira_context']}")
        if state.get("test_cases"):
            sources.append(f"\nGenerated Test Cases:\n{state['test_cases']}")
    
    # Combine context into final answer
    if sources:
        state["final_answer"] = "\n\n".join(sources)
    else:
        state["final_answer"] = "No relevant information found from enabled modules."
    
    # Save response to output directory
    if state["final_answer"]:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine module type
        module_type = "unknown"
        if module_manager.is_enabled('rag') and state.get("rag_context"):
            module_type = "rag"
        elif module_manager.is_enabled('sql') and state.get("sql_context"):
            module_type = "sql"
        elif module_manager.is_enabled('search') and state.get("serp_context"):
            module_type = "search"
        elif module_manager.is_enabled('jira') and state.get("jira_context"):
            module_type = "jira"
        
        # Save response
        filename = f"{module_type}_response_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Question: {state['input']}\n\n")
            f.write(state["final_answer"])
    
    return state

def should_use_jira(state: ChatState) -> bool:
    """Determine if JIRA node should be used."""
    return jira_config.is_configured()

def has_jira_results(state: ChatState) -> bool:
    """Check if JIRA results are present."""
    return bool(state.get("jira_context"))

def get_next_node(state: ChatState) -> str:
    """Determine which node to process next based on enabled modules and current node."""
    current_node = state.get("current_node", "entry")
    
    if current_node == "entry":
        if module_manager.is_enabled('rag'):
            state["current_node"] = "RAG"
            return "RAG"
        if module_manager.is_enabled('sql'):
            state["current_node"] = "MySQL"
            return "MySQL"
        if module_manager.is_enabled('search'):
            state["current_node"] = "WebSearch"
            return "WebSearch"
        if module_manager.is_enabled('jira'):
            state["current_node"] = "JIRA"
            return "JIRA"
        return "Answer"
    
    # After any module, go directly to Answer
    return "Answer"

def entry_node(state: ChatState) -> ChatState:
    """Entry node that initializes state."""
    state["current_node"] = "entry"
    return state

# LangGraph flow
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("entry", entry_node)
graph.add_node("RAG", rag_node)
graph.add_node("MySQL", mysql_node)
graph.add_node("WebSearch", serp_node)
graph.add_node("JIRA", jira_node)
graph.add_node("TestCases", test_case_node)
graph.add_node("Answer", final_answer_node)

# Set entry point
graph.set_entry_point("entry")

# Add conditional edges from entry and modules
graph.add_conditional_edges(
    "entry",
    get_next_node,
    {
        "RAG": "RAG",
        "MySQL": "MySQL",
        "WebSearch": "WebSearch",
        "JIRA": "JIRA",
        "Answer": "Answer"
    }
)

# Add edges from modules to Answer
graph.add_edge("RAG", "Answer")
graph.add_edge("MySQL", "Answer")
graph.add_edge("WebSearch", "Answer")

# JIRA can optionally go through TestCases
graph.add_conditional_edges(
    "JIRA",
    has_jira_results,
    {True: "TestCases", False: "Answer"}
)
graph.add_edge("TestCases", "Answer")

# Set finish point
graph.set_finish_point("Answer")

# Compile the graph
app = graph.compile()
