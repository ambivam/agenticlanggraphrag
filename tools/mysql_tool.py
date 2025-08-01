from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain.agents import AgentExecutor, create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
import os

class SQLAgent:
    def __init__(self, agent_executor):
        self.agent_executor = agent_executor
        self.is_sql_agent = True
    
    def invoke(self, input_text):
        return self.agent_executor.invoke(input_text)

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
        
        # Create LLM
        llm = ChatOpenAI(temperature=0)
        
        # Create SQL agent
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True
        )
        
        # Wrap agent in our custom class
        return SQLAgent(agent_executor)
    except Exception as e:
        print(f"Error creating MySQL agent: {str(e)}")
        # Return a dummy agent that will inform the user about the database connection issue
        return lambda x: {"output": "Database connection is currently unavailable. Please try other sources."}
