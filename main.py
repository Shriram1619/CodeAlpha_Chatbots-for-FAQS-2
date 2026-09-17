import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# 1. FAQ Knowledge Base
faqs = [
    {
        "question": "What is the duration of the CodeAlpha internship?",
        "answer": "The CodeAlpha internship typically lasts for 4 weeks (1 month)."
    },
    {
        "question": "How do I submit my completed tasks?",
        "answer": "You need to push your source code to a public GitHub repository and submit the repo link along with a video demo link via the submission form."
    },
    {
        "question": "Will I get a certificate upon completion?",
        "answer": "Yes, you will receive a Certificate of Completion once you successfully submit and pass the required tasks."
    },
    {
        "question": "How many tasks do I need to complete?",
        "answer": "You are required to complete at least 2 to 3 assigned tasks to qualify for the completion certificate."
    },
    {
        "question": "Is the internship paid or unpaid?",
        "answer": "The internship program at CodeAlpha is an unpaid learning and development opportunity."
    },
    {
        "question": "What tools can I use for the tasks?",
        "answer": "You can use Python, Google Colab, VS Code, Jupyter Notebooks, Flutter, or any framework suitable for your assigned domain."
    }
]

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# 2. Text Preprocessing
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

processed_questions = [preprocess_text(q) for q in questions]

# 3. Fit TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)

# 4. Response Matching Function
def get_bot_response(user_query, confidence_threshold=0.25):
    processed_query = preprocess_text(user_query)
    query_vector = vectorizer.transform([processed_query])
    
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    if best_score < confidence_threshold:
        return "I'm sorry, I don't have an answer for that question. Could you try rephrasing?"
    
    return answers[best_match_idx]

# 5. Interactive Chat Loop
print("=" * 50)
print("     Welcome to the CodeAlpha FAQ Assistant!      ")
print("=" * 50)
print("Type 'exit' or 'quit' to stop chatting.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ['exit', 'quit', 'bye']:
        print("Bot: Thank you for chatting! Good luck with your internship!")
        break
    
    if not user_input.strip():
        continue
        
    response = get_bot_response(user_input)
    print(f"Bot: {response}\n")
