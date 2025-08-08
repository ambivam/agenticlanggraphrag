import os
from dotenv import load_dotenv
from pathlib import Path
from tools.file_upload import update_faiss_index

# Load environment variables
env_path = Path(__file__).parent / '.env'
print(f"Loading .env from: {env_path}")
load_dotenv(dotenv_path=env_path, override=True)

# Check OpenAI key
openai_key = os.getenv('OPENAI_API_KEY')
print(f"OpenAI key loaded: {'YES' if openai_key else 'NO'}")

if not openai_key:
    print("Error: OpenAI API key not found in environment variables")
    exit(1)

# Update FAISS index with the Word document
doc_path = os.path.join('data', 'Document 7.docx')
print(f"\nProcessing document: {doc_path}")
num_chunks, status = update_faiss_index([doc_path])
print(f"\nProcessed chunks: {num_chunks}")
print(f"Status: {status}")
