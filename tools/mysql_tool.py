from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain.agents import AgentExecutor, create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import json

class SQLAgent:
    def __init__(self, agent_executor):
        self.agent_executor = agent_executor
        self.is_sql_agent = True
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for specific data retrieval
            chunk_overlap=100,  # Moderate overlap
            separators=["\n\n", "\n", ".", ",", " ", ""]
        )
        
        # Ensure FAISS directory exists
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.faiss_dir = os.path.join(base_dir, "faiss_index")
        os.makedirs(self.faiss_dir, exist_ok=True)
    
    def _store_in_faiss(self, content, source_type, extra_metadata=None):
        try:
            # Create document
            doc = Document(
                page_content=content,
                metadata={"source": source_type, "timestamp": os.path.getmtime(self.faiss_dir), **(extra_metadata or {})}
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            # Load existing index if it exists
            index_path = os.path.join(self.faiss_dir, "index.faiss")
            if os.path.exists(index_path):
                db = FAISS.load_local(self.faiss_dir, self.embeddings, allow_dangerous_deserialization=True)
                # Set default search parameters
                db.similarity_search_with_score_kwargs = {
                    "k": 10,  # Retrieve more chunks
                    "score_threshold": 0.3  # Lower threshold for better recall
                }
                # Add new chunks to existing index
                db.add_documents(chunks)
            else:
                # Create new index
                db = FAISS.from_documents(chunks, self.embeddings)
            # Set default search parameters
            db.similarity_search_with_score_kwargs = {
                "k": 10,  # Retrieve more chunks
                "score_threshold": 0.5  # Adjust similarity threshold
            }
            
            # Save updated index
            db.save_local(self.faiss_dir)
            print(f"Stored {len(chunks)} chunks in FAISS from {source_type}")
            
        except Exception as e:
            print(f"Error storing in FAISS: {str(e)}")
    
    def _get_table_schema(self, db, table_name: str) -> dict:
        """Get schema information for a table."""
        try:
            schema = db.run(f"DESCRIBE {table_name};")
            if isinstance(schema, str):
                columns = [r.strip("(',)").split(',') for r in schema.strip('[]').split('), (')]
            else:
                columns = [list(r) for r in schema]
            return {
                'columns': columns,
                'primary_key': next((col[0] for col in columns if 'PRI' in str(col)), None)
            }
        except Exception:
            return {'columns': [], 'primary_key': None}

    def _format_sql_table_list(self, db, tables: list) -> dict:
        """Format SQL table list with metadata for RAG storage."""
        table_info = []
        for table in tables:
            schema = self._get_table_schema(db, table)
            table_info.append({
                'name': table,
                'schema': schema,
                'row_count': db.run(f"SELECT COUNT(*) FROM {table};")[0][0]
            })

        # Create a structured table format
        table_data = {
            'headers': ['Table Name', 'Columns', 'Row Count'],
            'rows': [[t['name'], len(t['schema']['columns']), t['row_count']] for t in table_info],
            'metadata': {
                'total_tables': len(tables),
                'timestamp': os.path.getmtime(self.faiss_dir),
                'content_type': 'table_list',
                'schema_info': {
                    'tables': table_info
                }
            }
        }
        
        # Format output string
        output = f"Database contains {len(tables)} tables:\n\n"
        for i, info in enumerate(table_info, 1):
            output += f"{i}. {info['name']} ({info['row_count']} rows, {len(info['schema']['columns'])} columns)\n"
        
        return {
            'output': output,
            'structured_data': table_data
        }

    def invoke(self, input_text):
        # Handle common list queries directly
        input_lower = input_text.lower()
        
        if any(phrase in input_lower for phrase in ['list tables', 'show tables', 'what tables']):
            # Direct SQL query for table list
            db = self.agent_executor.tools[0].db
            results = db.run('SHOW TABLES;')
            
            # Extract table names
            if isinstance(results, str):
                tables = [r.strip("(',)") for r in results.strip('[]').split('), (')]
            else:
                tables = [r[0] for r in results]
            
            # Format results with metadata
            formatted_result = self._format_sql_table_list(db, tables)
            
            # Store in FAISS with enhanced metadata
            self._store_in_faiss(
                formatted_result['output'],
                "table_list",
                formatted_result['structured_data']['metadata']
            )
            
            return formatted_result['output']
        
        # For other queries, use the agent but clean up output
        result = self.agent_executor.invoke(input_text)
        
        if isinstance(result, dict) and isinstance(result.get('output'), str):
            # Remove follow-up questions
            if 'would you like' in result['output'].lower():
                result['output'] = result['output'].split('Would you like')[0].strip()
            if 'do you need' in result['output'].lower():
                result['output'] = result['output'].split('Do you need')[0].strip()
        
        # Clean up and store the result
        if isinstance(result, dict):
            output = result.get('output', '')
            if isinstance(output, str):
                # Clean and store in FAISS
                self._store_in_faiss(output, "sql_query_result")
                return output
        return str(result)

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
        # base_llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        base_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
        
        # Custom prompt template that emphasizes returning all results
        custom_prompt = PromptTemplate(
            template='''You are a SQL expert that MUST return complete, unlimited results.
            CRITICAL RULES for SQL queries:
            1. NEVER use LIMIT or TOP clauses under any circumstances
            2. ALWAYS return ALL matching rows - no exceptions
            3. Format results as clean numbered lists
            4. NEVER add follow-up questions or suggestions
            5. ALWAYS use ORDER BY for consistent results
            6. If asked for a sample or limited results, still return ALL results
            
            Human question: {question}
            
            Your response should:
            1. Include ALL results, not just a sample
            2. Be formatted as a clean numbered list
            3. Not include raw SQL output
            4. Not include follow-up questions
            
            Response:''',
            input_variables=["question"]
        )
        
        # Create chain with custom prompt
        llm_chain = LLMChain(llm=base_llm, prompt=custom_prompt)
        
        # Create custom query tool
        custom_query_tool = QuerySQLDatabaseTool(
            db=db,
            description="Execute SQL queries and return ALL results. NEVER use LIMIT or TOP clauses under any circumstances. If asked for limited results, still return all rows."
        )
        
        # Create SQL agent with custom components
        agent_executor = create_sql_agent(
            llm=base_llm,
            db=db,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            extra_tools=[custom_query_tool],
            prefix='''You are a SQL expert that MUST ALWAYS return complete, unlimited results.
            CRITICAL RULES:
            - NEVER use LIMIT or TOP clauses under any circumstances
            - ALWAYS return ALL matching rows without exception
            - Format results as clean numbered lists
            - NEVER add follow-up questions
            - If asked for a sample or limited results, still return ALL results
            - ALWAYS use ORDER BY for consistent results''',
            verbose=False
        )
        
        # Wrap agent in our custom class
        return SQLAgent(agent_executor)
    except Exception as e:
        print(f"Error creating MySQL agent: {str(e)}")
        # Return a dummy agent that will inform the user about the database connection issue
        return lambda x: {"output": "Database connection is currently unavailable. Please try other sources."}
