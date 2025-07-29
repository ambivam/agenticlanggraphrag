import os
from langchain.tools import Tool
from serpapi import GoogleSearch

def get_serp_tool():
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
                    # Get first result
                    first_result = results["organic_results"][0]
                    snippet = first_result.get("snippet", "")
                    if snippet:
                        return snippet
                return None
            except Exception as e:
                print(f"Error in SerpAPI search: {str(e)}")
                return None
        
        # Create tool
        return Tool(
            name="web_search",
            description="Search the web for current information.",
            func=search_with_error_handling,
            return_direct=True
        )
    except Exception as e:
        print(f"Error creating SerpAPI tool: {str(e)}")
        return None
