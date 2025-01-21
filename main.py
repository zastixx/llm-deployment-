# Import necessary Firebase packages
import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import initialize_app
import tempfile
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_together import Together
from dotenv import load_dotenv, find_dotenv

def init_firebase():
    """Initialize Firebase with the provided configuration."""
    config = {
        "type": "service_account",
        "project_id": "your-llm-maker",
        "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.environ.get("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
        "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_CERT_URL")
    }
    
    if not firebase_admin._apps:
        cred = credentials.Certificate(config)
        initialize_app(cred, {
            'storageBucket': "your-llm-maker.firebasestorage.app",
            'databaseURL': "https://your-llm-maker-default-rtdb.asia-southeast1.firebasedatabase.app"
        })

def download_faiss_index():
    """Download FAISS index from Firebase Storage."""
    bucket = storage.bucket()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Download index files
        index_files = ['index.faiss', 'index.pkl']
        local_paths = {}
        
        for file_name in index_files:
            blob = bucket.blob(f'data_byte/{file_name}')
            local_path = os.path.join(temp_dir, file_name)
            blob.download_to_filename(local_path)
            local_paths[file_name] = local_path
        
        print(f"Successfully downloaded FAISS index files to {temp_dir}")
        return temp_dir, local_paths
    
    except Exception as e:
        print(f"Error downloading FAISS index: {str(e)}")
        raise

def initialize_qa_system():
    """Initialize the QA system with Firebase-loaded FAISS index."""
    # Initialize Firebase
    init_firebase()
    
    # Download FAISS index
    temp_dir, local_paths = download_faiss_index()
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1-ablated",
        model_kwargs={"trust_remote_code": True}
    )
    
    # Load FAISS from temporary directory
    db = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    # Define prompt template
    prompt_template = """<s>[INST]This is a chat template, your name is VX1000 BetaV Model.
    This model is made by Tarun, trained by Tarun, and you have 1 billion parameters.
    Your primary objective is to provide accurate and concise information related to code and question solving.
    QUESTION: {question}
    CONTEXT: {context}
    CHAT HISTORY: {chat_history}[/INST]
    ASSISTANT:
    </s>"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['question', 'context', 'chat_history']
    )
    
    # Initialize conversation state
    conversation_state = {
        "messages": [],
        "memory": ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
    }
    
    # Initialize LLM
    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.7,
        max_tokens=1024,
        top_k=1,
        together_api_key=os.environ.get('T_API')
    )
    
    # Create QA chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=conversation_state["memory"],
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    
    return qa, conversation_state

# Create a requirements.txt file
def create_requirements():
    requirements = """
firebase-admin==6.2.0
python-dotenv==1.0.0
langchain==0.1.0
faiss-cpu==1.7.4
sentence-transformers==2.2.2
langchain-together==0.0.3
langchain-community==0.0.13
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())

if __name__ == "__main__":
    # Load environment variables
    load_dotenv(find_dotenv())
    
    # Initialize QA system
    qa, conversation_state = initialize_qa_system()
    
    print("Welcome to VX1000 BetaV! Type your question or 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Generate response
        print("Assistant is thinking...\n")
        result = qa.invoke(input=user_input)
        
        # Display response
        full_response = "⚠️ **_Note: Information provided may be inaccurate._**\n\n"
        full_response += result["answer"]
        print(f"Assistant: {full_response}\n")
        
        # Update conversation state
        conversation_state["messages"].extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": result["answer"]}
        ])
