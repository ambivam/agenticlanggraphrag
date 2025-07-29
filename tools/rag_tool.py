import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

def get_rag_chain():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    faiss_dir = os.path.join(base_dir, "faiss_index")
    
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
    return qa_chain
