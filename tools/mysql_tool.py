from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain.agents import AgentExecutor, create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

class SQLAgent:
    def __init__(self, agent_executor):
        self.agent_executor = agent_executor
        self.is_sql_agent = True
    
    def invoke(self, input_text):
        # Handle common list queries directly
        input_lower = input_text.lower()
        
        if 'list' in input_lower and 'country' in input_lower and 'name' in input_lower:
            # Direct SQL query for all countries
            db = self.agent_executor.tools[0].db
            results = db.run('SELECT name FROM country ORDER BY name;')
            
            # Format results as a clean numbered list
            if isinstance(results, str):
                countries = [r.strip("(',)") for r in results.strip('[]').split('), (')]
            else:
                countries = [r[0] for r in results]
                
            output = "Here is a complete list of all countries:\n\n"
            output += '\n'.join(f"{i+1}. {country}" for i, country in enumerate(countries))
            
            return {"output": output}
        
        # For other queries, use the agent but clean up output
        result = self.agent_executor.invoke(input_text)
        
        if isinstance(result, dict) and isinstance(result.get('output'), str):
            # Remove follow-up questions
            if 'would you like' in result['output'].lower():
                result['output'] = result['output'].split('Would you like')[0].strip()
            if 'do you need' in result['output'].lower():
                result['output'] = result['output'].split('Do you need')[0].strip()
        
        return result

def get_mysql_agent():
    try:
        # Get MySQL configuration from environment variables
        user = os.getenv('MYSQL_USER')
        password = os.getenv('MYSQL_PASSWORD')
        host = os.getenv('MYSQL_HOST')
        database = os.getenv('MYSQL_DATABASE')
        
        # Ensure all required variables are present
        if not all([user, password, host, database]):
            raise ValueError("Missing required MySQL environment variables")
            
        # Create connection string with proper escaping
        from urllib.parse import quote_plus
        password = quote_plus(password)  # Escape special characters in password
        db_uri = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        
        # Create database connection
        db = SQLDatabase.from_uri(db_uri)
        
        # Create LLM with custom prompt
        base_llm = ChatOpenAI(temperature=0)
        
        # Custom prompt template that emphasizes returning all results
        custom_prompt = PromptTemplate(
            template="""You are a SQL expert that always returns complete results without limiting them.
            When writing SQL queries:
            1. NEVER use LIMIT clause unless explicitly asked
            2. Always return ALL matching rows
            3. Format results as clean numbered lists
            4. Do not add follow-up questions or suggestions
            5. For list/show queries, use ORDER BY for consistent results
            
            Human question: {question}
            
            Your response should:
            1. Include ALL results, not just a sample
            2. Be formatted as a clean numbered list
            3. Not include raw SQL output
            4. Not include follow-up questions
            
            Response:""",
            input_variables=["question"]
        )
        
        # Create chain with custom prompt
        llm_chain = LLMChain(llm=base_llm, prompt=custom_prompt)
        
        # Create custom query tool
        custom_query_tool = QuerySQLDatabaseTool(
            db=db,
            description="Execute SQL queries and return ALL results without any LIMIT clause."
        )
        
        # Create SQL agent with custom components
        agent_executor = create_sql_agent(
            llm=base_llm,
            db=db,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            extra_tools=[custom_query_tool],
            prefix="""You are a SQL expert that always returns complete results.
            Never use LIMIT in queries unless explicitly asked.
            Always return ALL matching rows.
            Format results as clean numbered lists.
            Do not add follow-up questions.""",
            verbose=True
        )
        
        # Wrap agent in our custom class
        return SQLAgent(agent_executor)
    except Exception as e:
        print(f"Error creating MySQL agent: {str(e)}")
        # Return a dummy agent that will inform the user about the database connection issue
        return lambda x: {"output": "Database connection is currently unavailable. Please try other sources."}
