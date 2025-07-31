import os
from dotenv import load_dotenv
from typing import TypedDict, Optional, List, Dict, Any

from tools.rag_tool import get_rag_chain
from tools.mysql_tool import get_mysql_agent
from tools.serpapi_tool import get_serp_tool
from tools.jira_tool import search_jira_issues, jira_config
from tools.test_case_generator import generate_test_cases

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
rag_chain = get_rag_chain()
mysql_agent = get_mysql_agent()
serp_tool = get_serp_tool()

def rag_node(state: ChatState) -> ChatState:
    # Initialize state fields
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
        
    response = rag_chain.invoke(state["input"])
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
    try:
        # Run MySQL agent and get response
        response = mysql_agent.run(state["input"])
        
        # Clean up the result
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
    if not serp_tool:  # If tool wasn't created successfully
        state["serp_context"] = None
        state["found"] = False
        return state
        
    try:
        response = serp_tool.run(state["input"])
        if response:  # Only set context if we got a valid response
            state["serp_context"] = response
            state["found"] = True
        else:
            state["serp_context"] = None
            state["found"] = False
    except Exception as e:
        print(f"SerpAPI Error: {str(e)}")
        state["serp_context"] = None
        state["found"] = False
    return state

def jira_node(state: ChatState) -> ChatState:
    # Initialize state fields if not present
    if "jira_context" not in state:
        state["jira_context"] = None
    if "found" not in state:
        state["found"] = False
        
    if not jira_config.is_configured():
        state["jira_context"] = None
        state["found"] = False
        return state
        
    try:
        response = search_jira_issues(state["input"])
        if response:  # Only set context if we got a valid response
            state["jira_context"] = response
            state["found"] = True
        else:
            state["jira_context"] = None
            state["found"] = False
    except Exception as e:
        print(f"JIRA Error: {str(e)}")
        state["jira_context"] = None
        state["found"] = False
    return state

def test_case_node(state: ChatState) -> ChatState:
    """Generate test cases for JIRA issues."""
    if "test_cases" not in state:
        state["test_cases"] = None
        
    if state.get("jira_context"):
        # Check if this is a continuation request
        continue_previous = False
        query = state.get("query", "").lower()
        if any(x in query for x in ["more", "additional", "continue", "generate more"]):
            continue_previous = True
        
        state["test_cases"] = generate_test_cases(
            jira_content=state["jira_context"],
            temperature=state.get("llm_temperature", 0.7),
            scenarios_per_category=state.get("scenarios_per_category", 10),
            continue_previous=continue_previous
        )
    return state

def final_answer_node(state: ChatState) -> ChatState:
    # Initialize final answer
    state['final_answer'] = ""
    sources = []
    
    # Check RAG result
    if state.get('rag_context') and state.get('rag_context') != 'None':
        sources.append(f"From knowledge base: {state['rag_context']}")
    
    # Check MySQL result
    if state.get('sql_context') and state.get('sql_context') != 'None':
        sources.append(f"From database: {state['sql_context']}")
    
    # Check web search result
    if state.get('serp_context') and state.get('serp_context') != 'None':
        sources.append(f"From web search: {state['serp_context']}")
        
    # Check JIRA result and test cases
    if state.get('jira_context') and state.get('jira_context') != 'None':
        sources.append(f"From JIRA: {state['jira_context']}")
        if state.get('test_cases'):
            sources.append(f"\n\nGenerated Test Cases:\n{state['test_cases']}")
    
    # Combine all found sources
    if sources:
        state['final_answer'] = "\n".join(sources)
    else:
        state['final_answer'] = "I could not find relevant information from any of the available sources (knowledge base, database, or web search)."
    
    return state

def should_use_jira(state: ChatState) -> bool:
    """Determine if JIRA node should be used."""
    return jira_config.is_configured()

def has_jira_results(state: ChatState) -> bool:
    """Check if JIRA results are present."""
    return bool(state.get("jira_context"))

# LangGraph flow
graph = StateGraph(ChatState)

# Set entry point - always start with RAG
graph.set_entry_point("RAG")

# Add nodes
graph.add_node("RAG", rag_node)
graph.add_node("TestCases", test_case_node)
graph.add_node("MySQL", mysql_node)
graph.add_node("WebSearch", serp_node)
graph.add_node("JIRA", jira_node)
graph.add_node("Answer", final_answer_node)

# Define edges for sequential flow:
# RAG -> MySQL -> WebSearch -> [JIRA] -> Answer

# After RAG, always try MySQL next
graph.add_edge("RAG", "MySQL")

# After MySQL, always try WebSearch next
graph.add_edge("MySQL", "WebSearch")

# After WebSearch, conditionally go to JIRA or Answer
graph.add_conditional_edges(
    "WebSearch",
    should_use_jira,
    {True: "JIRA", False: "Answer"}
)

# After JIRA, conditionally go to TestCases or Answer
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
