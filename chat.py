import random
import json
import os
import torch
import pprint
import pandas as pd
from model import NeuralNet
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from nltk_utils import bag_of_words, tokenize
import google.generativeai as genai

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

os.environ['OPENAI_API_KEY'] = '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'

genai.configure(api_key='AIzaSyAcdUSwM-3XLfon6PD6EBuZ9WZY0liwEeY')

model_gen = genai.GenerativeModel('gemini-pro')
chat = model_gen.start_chat(history=[])

df = pd.read_csv('Engagement.csv')

llm = ChatOpenAI(model="gpt-4",temperature=0)

agent_executor = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type = "tool-calling",
    verbose = False,
    allow_dangerous_code=True
)

def get_gemini_response(msg):
    
    response = chat.send_message(msg)
    return response.text

def get_response(msg):
    if "csv" in msg.lower() or "engagement" or "engaged" in msg.lower():
        return agent_executor.run(msg)
    # sentence = tokenize(msg)
    # X = bag_of_words(sentence, all_words)
    # X = X.reshape(1, X.shape[0])
    # X = torch.from_numpy(X).to(device)

    # output = model(X)
    # _, predicted = torch.max(output, dim=1)

    # tag = tags[predicted.item()]

    # probs = torch.softmax(output, dim=1)
    # prob = probs[0][predicted.item()]
    # if prob.item() > 0.75:
    #     for intent in intents['intents']:
    #         if tag == intent["tag"]:
    #             return random.choice(intent['responses'])
    
    response = get_gemini_response(msg)
    return response


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

