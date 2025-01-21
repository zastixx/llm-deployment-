from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_together import Together
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Initialize FastAPI app
app = FastAPI()

# Initialize conversation state
conversation_state = {
    "messages": [],
    "memory": ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
}

# Load embeddings and retriever
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1-ablated",
    model_kwargs={"trust_remote_code": True}
)
db = FAISS.load_local("data_byte", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define the prompt template
prompt_template = """<s>[INST]This is a chat template, your name is VX1000 BetaV Model.
This model is made by Tarun, trained by Tarun, and you have 1 billion parameters.
Your primary objective is to provide accurate and concise information related to code and question solving.
QUESTION: {question}
CONTEXT: {context}
CHAT HISTORY: {chat_history}[/INST]
ASSISTANT:
</s>"""

prompt = PromptTemplate(template=prompt_template, input_variables=['question', 'context', 'chat_history'])

# Initialize the LLM
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=1024,
    top_k=1,
    together_api_key=os.environ.get('T_API', 'your_api_key_here')
)

# Define the QA system
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=conversation_state["memory"],
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Request model for a response
class Query(BaseModel):
    question: str

@app.post("/query")
async def query_rag(query: Query):
    user_input = query.question

    # Append user input to conversation
    conversation_state["messages"].append({"role": "user", "content": user_input})

    try:
        # Generate response
        result = qa.invoke(input=user_input)

        # Append assistant's response to conversation
        conversation_state["messages"].append({"role": "assistant", "content": result["answer"]})

        return {"response": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_conversation():
    """Reset the conversation state."""
    conversation_state["messages"] = []
    conversation_state["memory"].clear()
    return {"message": "Conversation reset successfully."}

@app.get("/")
async def root():
    return {"message": "Welcome to VX1000 BetaV RAG Model API"}
