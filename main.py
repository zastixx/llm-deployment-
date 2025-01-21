from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_together import Together
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize conversation state
conversation_state = {
    "messages": [],
    "memory": ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
}

# Initialize embeddings and retriever
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
    together_api_key=os.environ.get('T_API', '2042c792ea10764e6fdc41d0b039bbf0098b10fa74da17fe7942aceff43ba7bf')
)

# Define the QA system
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=conversation_state["memory"],
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    # Generate response
    result = qa.invoke(input=user_input)

    # Append to conversation state
    conversation_state["messages"].append({"role": "user", "content": user_input})
    conversation_state["messages"].append({"role": "assistant", "content": result["answer"]})

    # Return the assistant's response
    return jsonify({"response": result["answer"]})


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the conversation state."""
    conversation_state["messages"] = []
    conversation_state["memory"].clear()
    return jsonify({"message": "Conversation reset successfully."})


if __name__ == "__main__":
    app.run(debug=True)
