from typing import Optional, List, Dict
import pymysql
from pymysql import Error

class MySQLModule:
    def __init__(self):
        self.connection = None
        
    def initialize(self, host: str, user: str, password: str, database: str) -> bool:
        try:
            self.connection = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=database,
                cursorclass=pymysql.cursors.DictCursor
            )
            return True
        except Error as e:
            print(f"Error connecting to MySQL: {str(e)}")
            return False
            
    def execute_query(self, query: str) -> Optional[str]:
        try:
            if not self.connection or self.connection._closed:
                return "Error: Not connected to MySQL database"
                
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                
                # For SELECT queries
                if query.strip().upper().startswith("SELECT"):
                    results = cursor.fetchall()
                    if not results:
                        return "No results found"
                        
                    # Format results
                    output = []
                    for row in results:
                        output.append("\n".join(f"{k}: {v}" for k, v in row.items()))
                    return "\n\n".join(output)
                    
                # For other queries (INSERT, UPDATE, DELETE)
                else:
                    self.connection.commit()
                    return f"Query executed successfully. Affected rows: {cursor.rowcount}"
                    
        except Error as e:
            return f"Error executing query: {str(e)}"
                
    def close(self):
        if self.connection and not self.connection._closed:
            self.connection.close()
