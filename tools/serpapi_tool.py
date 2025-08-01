import os
from langchain.tools import Tool
from serpapi import GoogleSearch
from typing import List, Dict

def get_serp_tool():
    """Get the SerpAPI tool."""
    try:
        api_key = os.getenv('SERPAPI_API_KEY')
        if not api_key:
            raise ValueError("SERPAPI_API_KEY environment variable not set")
            
        def search_with_error_handling(query: str) -> str:
            try:
                # Create search parameters
                params = {
                    "q": query,
                    "api_key": api_key,
                    "engine": "google"
                }
                
                # Run search
                search = GoogleSearch(params)
                results = search.get_dict()
                
                # Extract organic results
                if "organic_results" in results and results["organic_results"]:
                    # Get top 3 results
                    search_results = []
                    for result in results["organic_results"][:3]:
                        title = result.get("title", "")
                        snippet = result.get("snippet", "")
                        link = result.get("link", "")
                        if snippet:
                            search_results.append(f"Source: {title}\nSummary: {snippet}\nLink: {link}\n")
                    
                    if search_results:
                        return "\n\n".join(search_results)
                return None
            except Exception as e:
                print(f"Error in SerpAPI search: {str(e)}")
                return None
        
        # Create tool
        tool = Tool(
            name="web_search",
            description="Search the web for current information.",
            func=search_with_error_handling,
            return_direct=True
        )
        
        # Add identifier
        tool.is_search_tool = True
        
        return tool
    except Exception as e:
        print(f"Error creating SerpAPI tool: {str(e)}")
        return None
