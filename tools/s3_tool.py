import os
from typing import TypedDict, Optional, List, Dict, Any, ClassVar
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

class S3Agent:
    def __init__(self, agent_executor):
        self.agent_executor = agent_executor
        self.is_s3_agent = True
    
    def invoke(self, input_text: str) -> str:
        return self.agent_executor.invoke({"input": input_text})

class ListBucketsTool(BaseTool):
    name: str = "list_buckets"
    description: str = "List all available S3 buckets in your AWS account"
    s3_client: Any = None

    def __init__(self):
        super().__init__()
        load_dotenv()
        
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region = os.getenv('AWS_REGION', 'us-east-1')
        
        if not access_key or not secret_key:
            raise ValueError("AWS credentials not found in environment variables")
            
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )

    def _run(self, tool_input: str) -> List[str]:
        try:
            response = self.s3_client.list_buckets()
            return [bucket['Name'] for bucket in response['Buckets']]
        except ClientError as e:
            return f"Error listing buckets: {str(e)}"

class ListFilesTool(BaseTool):
    name: str = "list_files"
    description: str = "List files in an S3 bucket with optional prefix"
    s3_client: Any = None

    def __init__(self):
        super().__init__()
        load_dotenv()
        
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region = os.getenv('AWS_REGION', 'us-east-1')
        
        if not access_key or not secret_key:
            raise ValueError("AWS credentials not found in environment variables")
            
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )

    def _run(self, tool_input: str) -> str:
        print(f"\n=== ListFilesTool Called ===\nInput: {tool_input}")
        try:
            # Parse input format: bucket_name[:filter]
            if isinstance(tool_input, dict):
                bucket_name = tool_input.get('bucket_name', '')
                file_filter = tool_input.get('filter', '')
            else:
                parts = tool_input.split(':', 1)
                bucket_name = parts[0].strip()
                file_filter = parts[1].strip() if len(parts) > 1 else ''
                
            print(f"\nAWS Credentials:")
            print(f"AWS_ACCESS_KEY_ID: {'✓ Set' if os.getenv('AWS_ACCESS_KEY_ID') else '✗ Not set'}")
            print(f"AWS_SECRET_ACCESS_KEY: {'✓ Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else '✗ Not set'}")
            print(f"AWS_REGION: {os.getenv('AWS_REGION', 'Not set')}")
            
            print(f"\nListing files:")
            print(f"Bucket: {bucket_name}")
            print(f"Filter: {file_filter}")
            
            print(f"Listing files in bucket: {bucket_name}, filter: {file_filter}")
            
            # List files in bucket
            try:
                print(f"Calling S3 list_objects_v2...")
                response = self.s3_client.list_objects_v2(Bucket=bucket_name)
                print(f"Response received")
                
                if 'Contents' not in response:
                    print("No files found in bucket")
                    return f"No files found in bucket '{bucket_name}'"
                    
                # Get all files
                files = [item['Key'] for item in response['Contents']]
                print(f"Found {len(files)} files:")
                for file in files:
                    print(f"- {file}")
                
                if 'Contents' not in response:
                    return f"No files found in bucket '{bucket_name}'"
                    
                # Get all files
                files = [item['Key'] for item in response['Contents']]
                
                # Apply filter if provided
                if file_filter:
                    print(f"\nApplying filter: {file_filter}")
                    files = [f for f in files if file_filter in f.lower()]
                    print(f"Found {len(files)} files after filtering")
                
                if not files:
                    msg = f"No files found in bucket '{bucket_name}'"
                    if file_filter:
                        msg += f" matching filter '{file_filter}'"
                    print(f"\nResult: {msg}")
                    return msg
                
                # Format response
                response = f"Found {len(files)} files in bucket '{bucket_name}':\n"
                response += "\n".join([f"  - {file}" for file in files])
                print(f"\nResult:\n{response}")
                return response
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'NoSuchBucket':
                    return f"Bucket '{bucket_name}' does not exist"
                elif error_code == 'AccessDenied':
                    return f"Access denied. Please check your IAM permissions for s3:ListBucket"
                else:
                    return f"AWS Error: {str(e)}"
        except Exception as e:
            return f"Error listing files: {str(e)}"

class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Read contents of a file from S3"
    s3_client: Any = None

    def __init__(self):
        super().__init__()
        load_dotenv()
        
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region = os.getenv('AWS_REGION', 'us-east-1')
        
        if not access_key or not secret_key:
            raise ValueError("AWS credentials not found in environment variables")
            
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )

    def _run(self, tool_input: str) -> str:
        try:
            # Parse input format: bucket_name:file_key
            if ':' not in tool_input:
                return "Please provide input in format 'bucket_name:file_key'"
                
            bucket_name, file_key = tool_input.split(':', 1)
            bucket_name = bucket_name.strip()
            file_key = file_key.strip()
            
            # Handle file key with or without extension
            if not file_key.endswith('.pdf') and not file_key.endswith('.txt'):
                # Try both .pdf and .txt extensions
                for ext in ['.pdf', '.txt']:
                    try:
                        response = self.s3_client.get_object(Bucket=bucket_name, Key=file_key + ext)
                        return f"Contents of {file_key + ext}:\n" + response['Body'].read().decode('utf-8')
                    except ClientError:
                        continue
                        
                # If no extension worked, try without extension
                try:
                    response = self.s3_client.get_object(Bucket=bucket_name, Key=file_key)
                    return f"Contents of {file_key}:\n" + response['Body'].read().decode('utf-8')
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchKey':
                        return f"File not found: {file_key} (tried with .pdf, .txt, and no extension)"
                    raise
            else:
                # Extension provided, try direct access
                response = self.s3_client.get_object(Bucket=bucket_name, Key=file_key)
                return f"Contents of {file_key}:\n" + response['Body'].read().decode('utf-8')
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                return f"Bucket '{bucket_name}' does not exist"
            elif error_code == 'NoSuchKey':
                return f"File not found in bucket '{bucket_name}'"
            elif error_code == 'AccessDenied':
                return f"Access denied. Please check your IAM permissions for s3:GetObject"
            else:
                return f"AWS Error: {str(e)}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

class SearchFilesTool(BaseTool):
    name: str = "search_files"
    description: str = "Search for files in S3 bucket"
    s3_client: Any = None
    list_files_tool: Any = None
    
    def __init__(self):
        super().__init__()
        # Load environment variables
        load_dotenv()
        
        # Get AWS credentials
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region = os.getenv('AWS_REGION', 'us-east-1')
        
        if not access_key or not secret_key:
            raise ValueError("AWS credentials not found in environment variables")
            
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        self.list_files_tool = ListFilesTool()
        super().__init__()

    def _run(self, tool_input: str) -> List[str]:
        try:
            # Parse input format: bucket_name:search_term
            if ':' not in tool_input:
                return ["Please provide input in format 'bucket_name:search_term'"]
                
            bucket_name, search_term = tool_input.split(':', 1)
            bucket_name = bucket_name.strip()
            search_term = search_term.strip()
            
            # List all files and filter by search term
            matching_files = self.list_files_tool._run(f"{bucket_name}:")
            if isinstance(matching_files, str) and 'Error' in matching_files:
                return [matching_files]
            
            # Filter files containing search term
            matching_files = [f for f in matching_files if search_term.lower() in f.lower()]
            if not matching_files:
                return [f"No files found containing '{search_term}' in bucket '{bucket_name}'"]
            
            # Read contents of matching files
            results = []
            for file_key in matching_files:
                content = self.s3_client.get_object(Bucket=bucket_name, Key=file_key)['Body'].read().decode('utf-8')
                results.append(f"File: {file_key}\nContents:\n{content}\n---")
            
            return results
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                return [f"Bucket '{bucket_name}' does not exist"]
            elif error_code == 'NoSuchKey':
                return [f"File not found in bucket '{bucket_name}'"]
            elif error_code == 'AccessDenied':
                return [f"Access denied. Please check your IAM permissions for s3:GetObject"]
            else:
                return [f"AWS Error: {str(e)}"]
        except Exception as e:
            return [f"Error searching files: {str(e)}"]



def get_s3_agent() -> S3Agent:
    """Create and return an S3 agent with natural language processing capabilities"""
    # Check for required environment variables
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Initialize tools
    list_buckets = ListBucketsTool()
    list_files = ListFilesTool()
    read_file = ReadFileTool()
    search_files = SearchFilesTool()
    
    # Create a wrapper function to route requests to the right tool
    def route_s3_request(inputs):
        print("\n=== S3 Agent Routing ===")
        print(f"Received inputs: {inputs}")
        try:
            # Get input and bucket from context
            input_text = inputs.get('input', '')
            bucket = inputs.get('bucket', '')
            
            print(f"\nParsing request:")
            print(f"- Input text: {input_text}")
            print(f"- Bucket: {bucket}")
            
            if not bucket:
                print("Error: No bucket specified")
                return "Error: No bucket specified"
                
            query = input_text.strip().lower()
            print(f"- Normalized query: {query}")
            
            print("\nRouting query...")
            # Route based on query type
            if 'list' in query:
                print("Detected LIST operation")
                # List files command with optional filter
                filter_type = 'pdf' if 'pdf' in query else ''
                print(f"- Bucket: {bucket}")
                print(f"- Filter: {filter_type}")
                
                print("\nCalling ListFilesTool...")
                result = list_files._run({'bucket_name': bucket, 'filter': filter_type})
                print(f"ListFilesTool result: {result}")
                return result
            elif any(word in query for word in ['read', 'show', 'content', 'get']):
                # Read file command
                words = query.split()
                file_name = next((word for word in reversed(words) if '.pdf' in word or '.txt' in word), words[-1])
                full_query = f"{bucket}:{file_name}"
                print(f"Routing to read file with query: {full_query}")
                return read_file._run(full_query)
            else:
                # Try list files by default
                print(f"Default routing to list files for bucket: {bucket}")
                return list_files._run(bucket)
        except Exception as e:
            return f"Error processing S3 request: {str(e)}"
    
    # Create an agent that uses the router
    class DirectS3Agent:
        def __init__(self):
            self.is_s3_agent = True
            
        def invoke(self, inputs):
            result = route_s3_request(inputs)
            return {"input": inputs.get('input', ''), "final_answer": result}
    
    return DirectS3Agent()
