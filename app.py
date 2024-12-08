import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter

    # Inject custom CSS
    st.markdown("""
      <style>
    body {
        background-color: #333333;  /* Light background color */
        color: #333333;  /* Dark text color */
    }
    .stApp {
        background-color: #333333;  /* Light background for the app */
    }
    .stTextInput input {
        color: #333333;  /* Darker text input color */
        background-color: #ffffff;  /* White background for input fields */
    }
    .stTextArea textarea {
        color: #333333;  /* Darker textarea color */
        background-color: #ffffff;  /* White background for textarea */
    }
    .sidebar .sidebar-content {
        background-color: #0066cc;  /* College brand color */
        color: white;
    }
    .css-1d391kg {
        background-color: #0099cc;  /* Button color */
        color: white;
    }
    .stButton button {
        background-color: #007bff;  /* Button color */
        color: white;
    }
    h1, h2, h3 {
        color: #0066cc;  /* College brand color */
    }
    .st-bf {
        background-color: #a3b0c3;  /* Light gray background */
        color: white;
    }
    .st-d3 {
        background-color: #67a9d7;  /* Light blue button or section */
        color: white;
    }
    .css-ffhzg2 {
        padding: 10px;
    }
    .stMarkdown {
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Set the title for the page
    st.title("College Chatbot")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the College Chatbot. Ask any question related to the college.")

        # Check if the chat_log.csv file exists
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("This chatbot is designed to assist students with their college-related queries.")
        st.subheader("Features:")
        st.write("""
        - Provides answers to frequently asked questions about college.
        - Saves conversation history for review.
        - Built using Natural Language Processing (NLP) techniques.
        """)

if __name__ == '__main__':
    main()
