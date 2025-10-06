from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv
import requests, os, json

# --- Load environment variables ---
load_dotenv()

app = Flask(__name__)

# --- Perplexity API key ---
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")

# --- Sentiment analyzer ---
sentiment_analyzer = pipeline("sentiment-analysis")

# --- Load wellness tips ---
try:
    with open("wellness_tips.json", "r", encoding="utf-8") as f:
        wellness_tips = json.load(f)
except:
    wellness_tips = {
        "relaxation": ["Take deep breaths and stretch your body."],
        "motivation": ["Believe in yourself. You are stronger than you think."]
    }

# --- System prompt ---
system_prompt = """
You are Saathi, a caring and empathetic mental health chatbot.
Always respond in a supportive, non-judgmental, and concise way.
Validate user emotions and provide wellness tips when needed.
"""

# --- Conversation history ---
HISTORY_FILE = "chat_history.json"
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            chat_history = json.load(f)
    except:
        chat_history = []
else:
    chat_history = []

# --- Sentiment analysis ---
def get_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result.get("label", "NEUTRAL"), result.get("score", 0)

# --- Perplexity API call with proper alternation ---
def ask_perplexity(user_input):
    if not PERPLEXITY_KEY:
        return "(Offline Mode) API key not found."

    # Include last 5 messages and system prompt
    messages = [{"role": "system", "content": system_prompt}] + chat_history[-5:]
    messages.append({"role": "user", "content": user_input})

    # --- CLEAN MESSAGES TO ALTERNATE ---
    clean_messages = []
    for msg in messages:
        if not clean_messages or msg["role"] != clean_messages[-1]["role"]:
            clean_messages.append(msg)
        else:
            # Merge consecutive messages of same role
            clean_messages[-1]["content"] += "\n" + msg["content"]

    headers = {"Authorization": f"Bearer {PERPLEXITY_KEY}", "Content-Type": "application/json"}
    payload = {"model": "sonar-pro", "messages": clean_messages, "temperature": 0.7, "max_tokens": 250}

    try:
        response = requests.post("https://api.perplexity.ai/chat/completions",
                                 headers=headers, json=payload)
        if response.status_code == 200:
            ai_reply = response.json()["choices"][0]["message"]["content"]
            chat_history.append({"role": "assistant", "content": ai_reply})
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(chat_history, f, indent=2)
            return ai_reply
        else:
            return f"(Offline Mode) Perplexity API error: {response.text}"
    except Exception as e:
        return f"(Offline Mode) Could not connect to Perplexity: {e}"

# --- Chatbot response ---
def chatbot_response(user_input):
    sentiment, _ = get_sentiment(user_input)

    if sentiment == "NEGATIVE":
        prefix = "I hear you, that sounds tough. You are not alone. "
        tip = f"💡 Tip: {wellness_tips['relaxation'][0]}"
    elif sentiment == "POSITIVE":
        prefix = "That’s great to hear! Keep your positive energy going. "
        tip = f"💡 Tip: {wellness_tips['motivation'][0]}"
    else:
        prefix = "Thanks for sharing. I'm here for you. "
        tip = ""

    # Append user message to history
    chat_history.append({"role": "user", "content": user_input})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=2)

    ai_reply = ask_perplexity(user_input)
    return prefix + ai_reply[:500] + tip  # Limit long replies

# --- Flask routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "Please enter a message."})
    response = chatbot_response(user_input)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
