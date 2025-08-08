from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Load the FAISS index
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Try a query about Bhagavad Gita
query = "What is the Bhagavad Gita?"
results = db.similarity_search(query, k=2)

print(f"\nQuery: {query}")
print("\nResults:")
for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Content: {doc.page_content}")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
