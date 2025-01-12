import random
import json
import os
import torch
import google.generativeai as genai
import pandas as pd

from model import NeuralNet
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatGooglePaLM
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents for fallback response logic
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load pre-trained model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize the neural network
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

# Set up Google API Key for generative models
api_key = "AIzaSyAcdUSwM-3XLfon6PD6EBuZ9WZY0liwEeY"
genai.configure(api_key=api_key)
chat = genai.Chat(model="gemini-1")  # Replace with correct model

# LangChain integration with Google PaLM
llm = ChatGooglePaLM(api_key=api_key, model="chat-bison", temperature=0)

# Function to create a LangChain agent for Excel data
def create_excel_agent(file_path):
    """Create a LangChain agent for querying a Pandas DataFrame."""
    try:
        loader = UnstructuredExcelLoader(file_path)
        docs = loader.load()
        if not docs:
            return "No data found in the Excel file."
        df = pd.DataFrame([doc.metadata for doc in docs])
        if df.empty:
            return "Loaded Excel file is empty."
        return create_pandas_dataframe_agent(llm, df)
    except Exception as e:
        return f"Failed to create Excel agent: {str(e)}"

# Function to handle Gemini and Excel-based responses
def get_gemini_response(msg, excel_path=None):
    """Get a response from either Gemini or Excel agent."""
    if excel_path:
        # Attempt to query Excel agent
        agent = create_excel_agent(excel_path)
        if isinstance(agent, str):  # Error handling
            return agent
        try:
            response = agent.run(msg)
            return response
        except Exception as e:
            return f"Error while querying the Excel data: {str(e)}"

    # Fallback to Gemini for general queries
    response = chat.send_message(msg)
    if hasattr(response, 'text'):
        return response.text
    else:
        return "Failed to fetch a response from Gemini."

# Unified response function
def get_response(msg, excel_path=None):
    """Get a response using either Gemini or the Excel agent."""
    return get_gemini_response(msg, excel_path)

# Main chat loop
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    print("You can optionally provide an Excel file for data-driven queries.")
    
    excel_path = "C:\\Users\\HIMANGI\\chatbot-deployment\\Data.xlsx"
    if not os.path.isfile(excel_path):
        print("Excel file not found. Defaulting to general chat mode.")
        excel_path = None

    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        resp = get_response(sentence, excel_path)
        print(f"{bot_name}: {resp}")
