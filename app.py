from flask import Flask, render_template, request, jsonify
import openai
from transformers import pipeline
import json
import os

# --- Load OpenAI API key from environment variable ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Initialize Flask ---
app = Flask(__name__)

# --- Load sentiment analysis pipeline (HuggingFace) ---
sentiment_analyzer = pipeline("sentiment-analysis")

# --- Load wellness tips JSON safely ---
try:
    with open("wellness_tips.json", "r", encoding="utf-8") as f:
        wellness_tips = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    wellness_tips = {"relaxation": [], "motivation": []}

# --- Function: analyze sentiment ---
def get_sentiment(text: str):
    result = sentiment_analyzer(text)[0]
    return result.get("label", "NEUTRAL"), result.get("score", 0)

# --- Function: chatbot reply using new OpenAI Chat API ---
def chatbot_response(user_input: str) -> str:
    sentiment, _ = get_sentiment(user_input)
    prefix = "I hear you, that sounds tough. " if sentiment=="NEGATIVE" else "Thanks for sharing. "

    try:
        # Try calling OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a friendly mental health chatbot."},
                {"role": "user", "content": user_input}
            ]
        )
        reply = response.choices[0].message.content.strip()
    except:
        # Fallback reply if quota exceeded
        reply = "This is a local test response. GPT API quota exceeded."

    return prefix + reply


# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = data.get("message", "")
    if not user_input.strip():
        return jsonify({"reply": "Please enter a message."})
    response = chatbot_response(user_input)
    return jsonify({"reply": response})

# --- Run Flask app ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
