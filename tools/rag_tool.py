import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

class RAGTool:
    def __init__(self, chain):
        self.chain = chain
        self.is_rag_tool = True
    
    def invoke(self, input_text):
        try:
            print(f"\nRAG Tool - Processing query: {input_text}")
            
            # Get raw similarity search results first
            if hasattr(self.chain, 'retriever') and hasattr(self.chain.retriever, 'vectorstore'):
                print("\nPerforming similarity search...")
                similar_docs = self.chain.retriever.vectorstore.similarity_search_with_score(
                    input_text,
                    k=5
                )
                print("\nSimilarity search results:")
                for i, (doc, score) in enumerate(similar_docs, 1):
                    print(f"\nResult {i} (similarity score: {score:.4f}):")
                    print(f"Content preview: {doc.page_content[:200]}...")
                    print(f"Metadata: {doc.metadata}")
            
            # Now invoke the chain
            result = self.chain.invoke(input_text)
            print(f"\nRAG Tool - Got result: {result}")
            
            # Extract source documents and scores
            if hasattr(result, 'get') and result.get('source_documents'):
                print("\nSource documents used:")
                for i, doc in enumerate(result['source_documents'], 1):
                    print(f"\nDocument {i}:")
                    print(f"Content: {doc.page_content[:200]}...")
                    if hasattr(doc, 'metadata'):
                        print(f"Metadata: {doc.metadata}")
            
            return result
        except Exception as e:
            print(f"\nError in RAG Tool invoke: {str(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return None

def get_rag_chain():
    """Get the RAG chain."""
    try:
        print("\n=== Creating RAG Chain ===")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        faiss_dir = os.path.join(base_dir, "faiss_index")
        print(f"FAISS directory: {faiss_dir}")
        
        if not os.path.exists(faiss_dir):
            print(f"Error: FAISS directory not found at {faiss_dir}")
            return None
            
        index_path = os.path.join(faiss_dir, "index.faiss")
        pkl_path = os.path.join(faiss_dir, "index.pkl")
        
        if not (os.path.exists(index_path) and os.path.exists(pkl_path)):
            print(f"Error: Missing required files in {faiss_dir}")
            print(f"index.faiss exists: {os.path.exists(index_path)}")
            print(f"index.pkl exists: {os.path.exists(pkl_path)}")
            return None
        
        print("Loading OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()
        
        print("Loading FAISS index...")
        db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        
        # Debug: Print total documents in index
        try:
            doc_count = len(db.docstore._dict)
            print(f"Total documents in FAISS index: {doc_count}")
            print("\nSample document IDs and contents:")
            for doc_id in list(db.docstore._dict.keys())[:3]:  # Show first 3 docs
                doc = db.docstore._dict[doc_id]
                print(f"\nDoc ID: {doc_id}")
                print(f"Content preview: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
        except Exception as e:
            print(f"Error inspecting index: {str(e)}")
        
        print("\nCreating retriever...")
        retriever = db.as_retriever(
            search_type="mmr",  # Use MMR for diversity
            search_kwargs={
                "k": 5,  # Retrieve more documents
                "lambda_mult": 0.5,  # Balanced diversity and relevance
                "fetch_k": 10,  # Fetch more docs for better diversity
                "score_threshold": 0.3,  # Lower threshold to catch more matches
            }
        )
        # retriever = db.as_retriever(
        #     search_type="mmr",  # Use MMR for diversity
        #     search_kwargs={
        #         "k": 3,  # Reduced number of documents to retrieve
        #         "lambda_mult": 0.7,  # MMR diversity factor (0=max diversity, 1=max relevance)
        #         "fetch_k": 5,  # Fetch more docs then select k most diverse
        #         "score_threshold": 0.7,  # Increased similarity threshold for better relevance
        #     }
        # )

        
        print("Creating RAG chain...")
        # Custom prompt template for SQL queries
        from langchain.prompts import PromptTemplate
        custom_prompt = PromptTemplate(
            template="""You are a helpful assistant that provides information about SQL database contents.
            If the retrieved documents contain database query results, format them clearly.
            If you see table contents, list them in a clear, readable format.
            
            Question: {question}
            
            Context: {context}
            
            Answer: Let me help you with that information.""",
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                temperature=0.7,
                model="gpt-4"  # Use GPT-4 for better comprehension
            ),
            chain_type="stuff",  # Use stuff chain type for better context integration
            retriever=retriever,
            chain_type_kwargs={
                "prompt": custom_prompt
            },
            return_source_documents=True,
            verbose=True  # Add verbose output for debugging
        )
        
        print("Wrapping chain in RAGTool...")
        return RAGTool(chain)
    except Exception as e:
        print(f"\nError creating RAG chain: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return None
