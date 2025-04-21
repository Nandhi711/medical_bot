import os
from pinecone import Pinecone
from flask import Flask, render_template, jsonify, request
import pickle
import json
import numpy as np
import openai  # For Chatbot functionality
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Set environment variables for Pinecone and OpenAI API keys
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize Pinecone client (Updated method)
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# Load Hugging Face embeddings (Directly from Hugging Face)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Disease prediction model setup
with open('model/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/columns_encoded.json', 'r') as f:
    columns_encoding = json.load(f)

# List of symptoms from the encoding
symptoms_list = list(columns_encoding.keys())

# Setup Pinecone and LangChain for Chatbot (RAG)
index_name = "medicalbot"

# Use the new Pinecone instance to access the index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = OpenAI(temperature=0.4, max_tokens=500)

# Create a prompt that expects both 'input' and 'context'
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful medical assistant. The context is: {context}"),
        ("human", "{input}"),
    ]
)

# Create retrieval chain for question answering
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Route for Disease Prediction functionality
@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    if request.method == "POST":
        user_symptoms = request.form.getlist("symptoms")

        if not user_symptoms:
            return render_template("index.html", symptoms=symptoms_list, predictions="No symptoms selected")

        # Prepare input vector for the prediction model
        symptoms_input = [0] * len(columns_encoding)
        for symptom in user_symptoms:
            index = columns_encoding.get(symptom)
            if index is not None:
                symptoms_input[index] = 1

        # Predict the top 5 possible diseases
        symptoms_input = [symptoms_input]
        probs = model.predict_proba(symptoms_input)
        top_5_indices = np.argsort(probs[0])[-5:][::-1]
        top_predictions = [(model.classes_[idx], round(probs[0][idx] * 100, 2)) for idx in top_5_indices]

        predictions = top_predictions

    return render_template("index.html", symptoms=symptoms_list, predictions=predictions)

# Route for Chatbot functionality
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"response": "Please provide a message."})
    
    print(f"User message: {msg}")

    # Invoke the LangChain RAG chain with the message
    response = rag_chain.invoke({"input": msg, "context": "medical context"})

    # Return the chatbot response
    chatbot_response = response.get("answer", "Sorry, I couldn't understand that.")
    print(f"Chatbot response: {chatbot_response}")

    return jsonify({"response": chatbot_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
