import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

class RAGTool:
    def _build_context(self, current_query):
        """Build context from conversation history."""
        context = []
        for message in self.conversation_history:
            role = message.get("role", "")
            content = message.get("content", "")
            if role and content:
                context.append(f"{role.title()}: {content}")
        return "\n".join(context)
        
    def __init__(self, chain):
        self.chain = chain
        self.is_rag_tool = True
        self.conversation_history = []
    
    def invoke(self, input_text, thread_id=None):
        try:
            print(f"\nRAG Tool - Processing query: {input_text}")
            
            # Get conversation history from memory store
            from langgraph_mcp_bot import MemoryStore
            
            if thread_id:
                self.conversation_history = MemoryStore.get_memory(thread_id)
                # Add current query to memory
                MemoryStore.add_memory(thread_id, {"role": "user", "content": input_text})
            
            # Build context from conversation history
            context = self._build_context(input_text)
            
            # Debug: Print chain and retriever info
            print("\nChain type:", type(self.chain))
            print("Has retriever:", hasattr(self.chain, 'retriever'))
            if hasattr(self.chain, 'retriever'):
                print("Retriever type:", type(self.chain.retriever))
                print("Has vectorstore:", hasattr(self.chain.retriever, 'vectorstore'))
            
            # Get raw similarity search results
            if hasattr(self.chain, 'retriever') and hasattr(self.chain.retriever, 'vectorstore'):
                print("\nPerforming similarity search...")
                try:
                    similar_docs = self.chain.retriever.vectorstore.similarity_search_with_score(
                        input_text,
                        k=5
                    )
                    print("\nSimilarity search results:")
                    for i, (doc, score) in enumerate(similar_docs, 1):
                        print(f"\nResult {i} (similarity score: {score:.4f}):")
                        print(f"Content preview: {doc.page_content[:200]}...")
                        print(f"Metadata: {doc.metadata}")
                except Exception as e:
                    print(f"Error in similarity search: {str(e)}")
                    import traceback
                    print(f"Traceback:\n{traceback.format_exc()}")
            
            # Format chat history for chain input
            chat_history = [("Human", msg["content"]) if msg["role"] == "user" 
                           else ("Assistant", msg["content"]) 
                           for msg in self.conversation_history]
            
            # Debug chain info
            print("\nChain type:", type(self.chain))
            print("Chain components:", dir(self.chain))
            
            try:
                # Get raw similarity search results first
                if hasattr(self.chain, 'retriever'):
                    print("\nPerforming similarity search...")
                    docs = self.chain.retriever.get_relevant_documents(input_text)
                    print(f"Found {len(docs)} relevant documents")
                    for i, doc in enumerate(docs):
                        print(f"\nDocument {i+1}:")
                        print(f"Content: {doc.page_content[:200]}...")
                        print(f"Metadata: {doc.metadata}")
                
                # Invoke chain with formatted inputs
                chain_response = self.chain({
                    "question": input_text,
                    "chat_history": chat_history
                })
                
                print("\nRaw chain response:", chain_response)
                
                # Process and clean the response
                result = {}
                if isinstance(chain_response, dict):
                    # Extract and clean the answer
                    answer = chain_response.get('answer', '')
                    if not answer and 'result' in chain_response:
                        answer = chain_response['result']
                    
                    if answer:
                        # Clean special characters and normalize whitespace
                        answer = answer.replace('\xa0', ' ')
                        answer = answer.replace('\x0b', '\n')
                        answer = ' '.join(answer.split())
                        result['result'] = answer
                    
                    # Process source documents
                    if 'source_documents' in chain_response:
                        result['source_documents'] = chain_response['source_documents'][:3]
                else:
                    result = {'result': str(chain_response)}
            except Exception as e:
                print(f"Error in chain execution: {str(e)}")
                import traceback
                print(f"Traceback:\n{traceback.format_exc()}")
                result = {'error': str(e)}
            print(f"\nRAG Tool - Got result: {result}")
            
            # Extract source documents and scores
            if hasattr(result, 'get') and result.get('source_documents'):
                print("\nSource documents used:")
                for i, doc in enumerate(result['source_documents'], 1):
                    print(f"\nDocument {i}:")
                    print(f"Content: {doc.page_content[:200]}...")
                    if hasattr(doc, 'metadata'):
                        print(f"Metadata: {doc.metadata}")
            
            # Store assistant's response after getting it
            if thread_id and hasattr(result, 'get'):
                MemoryStore.add_memory(thread_id, {"role": "assistant", "content": str(result.get('result', ''))})
                
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
        data_dir = os.path.join(base_dir, "data")
        print(f"FAISS directory: {faiss_dir}")
        print(f"Data directory: {data_dir}")
        
        # Create embeddings
        print("Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        
        # Load or create FAISS index
        if os.path.exists(faiss_dir) and os.path.exists(os.path.join(faiss_dir, "index.faiss")):
            print("Loading existing FAISS index...")
            try:
                db = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
                print("Loaded existing index successfully")
            except Exception as e:
                print(f"Error loading existing index: {str(e)}")
                print("Will attempt to rebuild index")
                db = None
        else:
            print("No existing index found")
            db = None
            
        # If loading failed or no index exists, create new one
        if db is None:
            print("Creating new FAISS index...")
            try:
                # Load documents
                from langchain.document_loaders import DirectoryLoader
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                
                # Load all supported file types
                loader = DirectoryLoader(
                    data_dir,
                    glob="**/*.*",
                    show_progress=True,
                    use_multithreading=True
                )
                print("Loading documents...")
                documents = loader.load()
                print(f"Loaded {len(documents)} documents")
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                print("Splitting documents...")
                texts = text_splitter.split_documents(documents)
                print(f"Created {len(texts)} text chunks")
                
                # Create FAISS index
                print("Creating new vectorstore...")
                db = FAISS.from_documents(texts, embeddings)
                
                # Save index
                print("Saving new index...")
                os.makedirs(faiss_dir, exist_ok=True)
                db.save_local(faiss_dir)
                print("Saved new index successfully")
            except Exception as e:
                print(f"Error creating new index: {str(e)}")
                import traceback
                print(f"Traceback:\n{traceback.format_exc()}")
                return None
        
        # If we have a valid db, print stats
        if db is not None:
            try:
                doc_count = len(db.docstore._dict)
                print(f"\nTotal documents in index: {doc_count}")
                
                if doc_count > 0:
                    print("\nSample documents:")
                    for doc_id in list(db.docstore._dict.keys())[:2]:
                        doc = db.docstore._dict[doc_id]
                        print(f"\nDoc ID: {doc_id}")
                        print(f"Content preview: {doc.page_content[:200]}...")
                        print(f"Metadata: {doc.metadata}")
                else:
                    print("Warning: Index contains no documents!")
                    return None
            except Exception as e:
                print(f"Error inspecting index: {str(e)}")
                return None
        

        

        
        print("\nCreating retriever...")
        
        retriever = db.as_retriever(
            search_type="mmr",  # Use MMR for diversity
            search_kwargs={
                "k": 3,  # Reduced number of documents
                "lambda_mult": 0.7,  # Balance between relevance and diversity
                "fetch_k": 5,  # Fetch more docs for selection
                "score_threshold": 0.5,  # Higher threshold for better relevance
            }
        )
        
        print("Creating RAG chain...")
        # Custom prompt template for JIRA and SQL content
        from langchain.prompts import PromptTemplate
        custom_prompt = PromptTemplate(
            template='''You are a helpful assistant that provides well-structured answers with clear formatting.
            
            Previous conversation:
            {chat_history}
            
            Current question: {question}
            
            Context information:
            {context}
            
            Instructions:
            1. Start with a brief introduction paragraph
            2. Use ## for major sections (e.g. ## Overview)
            3. Leave a blank line before and after each section heading
            4. For each section:
               - Start with a brief section intro if needed
               - Use bullet points (* ) with a space after the asterisk
               - Bold (**key terms**) within the text
               - Use proper line breaks between bullets
            5. If information is missing, state so clearly in a separate paragraph
            
            Answer: Let me provide information about {question}:

'''
            
            
            ,
            input_variables=["context", "question", "chat_history"]
        )
        
        from langchain.chains import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferMemory
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        # Create the ConversationalRetrievalChain
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                temperature=0.7,
                model="gpt-4-turbo-preview",
                max_tokens=4000
            ),
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": custom_prompt,
                "document_prompt": PromptTemplate(
                    input_variables=["page_content"],
                    template="{page_content}\n"
                )
            },
            verbose=True
        )
        
        print("Wrapping chain in RAGTool...")
        return RAGTool(chain)
    except Exception as e:
        print(f"\nError creating RAG chain: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return None
