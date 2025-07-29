from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
import os

def ingest_docs():
    embeddings = OpenAIEmbeddings()
    docs = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"data/{file}")
        elif file.endswith(".txt"):
            loader = TextLoader(f"data/{file}")
        else:
            continue
        docs.extend(loader.load())

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index", allow_dangerous_deserialization=True)

if __name__ == "__main__":
    ingest_docs()
