import streamlit as st
import os
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import BartTokenizer, BartForConditionalGeneration
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import pandas as pd

# Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase_credentials.json")

# Check if Firebase Admin SDK has been initialized
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

class GroqChatAgent:
    def __init__(self, api_key, system_prompt, memory_length, csv_file):
        self.groq_api_key = api_key
        self.model = 'llama3-8b-8192'  # model
        self.system_prompt = system_prompt
        self.memory = ConversationBufferWindowMemory(k=memory_length, memory_key="chat_history", return_messages=True)
        self.groq_chat = ChatGroq(groq_api_key=self.groq_api_key, model_name=self.model)
        self.chat_history = []
        self.data = pd.read_csv(csv_file)

    def get_response(self, user_question):
        if not self.is_farming_related(user_question):
            if "give me news about" in user_question.lower():
                return self.handle_news_request(user_question)
            else:
                return "I can only answer farming-related questions. Please ask about farming."

        # Extracting insights from the CSV data
        csv_insight = self.extract_insight(user_question)

        for message in self.chat_history:
            self.memory.save_context(
                {'input': message['human']},
                {'output': message['AI']}
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(f"{user_question}\n\nData Insight: {csv_insight}")
            ]
        )

        conversation = LLMChain(
            llm=self.groq_chat,
            prompt=prompt,
            verbose=True,
            memory=self.memory,
        )

        response = conversation.predict(human_input=user_question)
        self.chat_history.append({'human': user_question, 'AI': response})
        return response

    def is_farming_related(self, question):
        farming_keywords = ["crop", "yield", "farm", "agriculture", "soil", "plant", "harvest", "irrigation", "pest", "fertilizer", "weather", "climate"]
        for keyword in farming_keywords:
            if keyword in question.lower():
                return True
        return False

    def extract_insight(self, question):
        if "crop" in question.lower():
            return self.data["Crop"].unique().tolist()
        elif "yield" in question.lower():
            return self.data[["Crop", "Yield"]].groupby("Crop").mean().to_dict()
        else:
            return "No specific insights found for this question."

    def fetch_latest_articles(self, limit=5):
        db = firestore.client()
        articles_ref = db.collection('articles').order_by('date', direction=firestore.Query.DESCENDING).limit(limit)
        docs = articles_ref.stream()

        articles_data = []
        for doc in docs:
            article = doc.to_dict()
            articles_data.append(article)
        return articles_data

    def summarize_article(self, article_text):
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        summarization_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        inputs = tokenizer(article_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = summarization_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
        summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summarized_text

    def handle_news_request(self, request):
        if "give me news about" in request.lower():
            topic = request.lower().split("give me news about")[-1].strip()
            print(f"Fetching news about {topic}...\n")

            latest_articles = self.fetch_latest_articles(limit=5)
            summarized_texts = []
            for article in latest_articles:
                summarized_text = self.summarize_article(article['text'])
                summarized_texts.append({
                    "date": article['date'],
                    "summary": summarized_text
                })

            response_text = ""
            for summarized_article in summarized_texts:
                response_text += f"Date: {summarized_article['date']}\n"
                response_text += f"Summarized Text:\n{summarized_article['summary']}\n\n"
            return response_text

        else:
            return "Please provide a valid request in the format 'Give me news about {topic}'."

# Config
groq_api_key = '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'
system_prompt = "You're an AI bot here to help farmers, so restrict all your responses to farming-related practices. be kind and humble to the farmers, make sure you're understanding and give kind replies without any slurs or bad language. Do not bring up any ill omens."
memory_length = 5
csv_file = 'Crop_Recommendation.csv'

#Initialization
agent = GroqChatAgent(groq_api_key, system_prompt, memory_length, csv_file)

# Streamlit
st.title("I'm Keithi!")
st.write("Your friendly Farming Assistant!")

user_question = st.text_input("Ask me a question:")

if user_question:
    response = agent.get_response(user_question)
    st.write("Keithi:", response)
