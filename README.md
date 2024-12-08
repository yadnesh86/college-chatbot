Chatbot using NLP
Overview
This project implements a chatbot using Natural Language Processing (NLP) techniques. The chatbot is designed to understand user intents and provide appropriate responses based on predefined patterns and responses. It utilizes the nltk library for natural language processing, scikit-learn for machine learning, and streamlit for creating an interactive web interface.

Features
Understands various user intents such as greetings, farewells, gratitude, and more.
Provides relevant responses based on user input.
Maintains a conversation history that can be viewed by the user.
Built using Python and leverages popular libraries for NLP and machine learning.
Technologies Used
Python
NLTK
Scikit-learn
Streamlit
JSON for intents data
Installation
1. Clone the Repository
git clone <repository-url>
cd <repository-directory>
2. Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install Required Packages
pip install -r requirements.txt
4. Download NLTK Data
import nltk
nltk.download('punkt')
Usage
To run the chatbot application, execute the following command:

streamlit run app.py
Once the application is running, you can interact with the chatbot through the web interface. Type your message in the input box and press Enter to see the chatbot's response.

Intents Data
The chatbot's behavior is defined by the intents.json file, which contains various tags, patterns, and responses. You can modify this file to add new intents or change existing ones.

