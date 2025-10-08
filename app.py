from flask import Flask, render_template, request, jsonify
import json, os, requests
from dotenv import load_dotenv
import re

# --- Load environment variables ---
load_dotenv()
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")

# --- Initialize Flask ---
app = Flask(__name__)

# --- Chat history ---
HISTORY_FILE = "chat_history.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        chat_history = json.load(f)
else:
    chat_history = []

# --- Wellness tips ---
wellness_tips = {
    "relaxation": [
        "Take 5 deep breaths and stretch your arms.",
        "Try a 2-minute meditation break.",
        "Write down one thing you’re grateful for."
    ],
    "motivation": [
        "You are stronger than you think.",
        "Every step forward is progress.",
        "It’s okay to rest; you don’t have to do everything at once."
    ]
}

# --- System prompt ---
system_prompt = """
You are Saathi, a friendly, caring, and empathetic mental health chatbot.
Always respond in a supportive, concise, and non-judgmental way.
"""

# --- Format AI reply into structured HTML ---
def format_reply(ai_text):
    lines = ai_text.split("\n")
    bullets = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "*")):
            bullets.append(f"<li>{line[1:].strip()}</li>")
        else:
            sentences = re.split(r'(?<=[.!?]) +', line)
            for sent in sentences:
                if sent.strip():
                    bullets.append(f"<li>{sent.strip()}</li>")

    if bullets:
        html = "<ul>\n" + "\n".join(bullets) + "\n</ul>"
    else:
        html = f"<p>{ai_text}</p>"

    return html

# --- Chatbot using Perplexity ---
def ask_perplexity(user_input):
    if not PERPLEXITY_KEY:
        return "(Offline Mode) API key not set."

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-10:]:
        if msg["role"] in ["user", "assistant"]:
            messages.append(msg)
    messages.append({"role": "user", "content": user_input})

    headers = {"Authorization": f"Bearer {PERPLEXITY_KEY}", "Content-Type": "application/json"}
    payload = {"model": "sonar-pro", "messages": messages, "temperature": 0.7, "max_tokens": 250}

    try:
        r = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload)
        if r.status_code == 200:
            result = r.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "(Offline Mode) Perplexity API returned no choices."
        else:
            print("Perplexity API error:", r.status_code, r.text)
            return f"(Offline Mode) Perplexity API error: {r.status_code}"
    except Exception as e:
        print("Perplexity API exception:", e)
        return "(Offline Mode) Could not connect to Perplexity API."

# --- Chatbot response ---
def chatbot_response(user_input):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

    ai_reply = ask_perplexity(user_input)
    ai_reply_structured = format_reply(ai_reply)

    # If greeting, only show AI encouragement
    if user_input.lower().strip() in greetings:
        structured_reply = f"""
        <div class="saathi-response">
            <h3>✨ Encouragement</h3>
            {ai_reply_structured}
        </div>
        """
    else:
        prefix = "😊 Feedback: Thanks for sharing. "
        tips = wellness_tips['motivation'][:2]

        structured_reply = f"""
        <div class="saathi-response">
            <h3>😊 Feedback</h3>
            <p>{prefix}</p>

            <h3>📝 Tips</h3>
            <ul>
        """
        for tip in tips:
            structured_reply += f"<li>{tip}</li>"
        structured_reply += "</ul>"

        structured_reply += f"""
            <h3>✨ Encouragement</h3>
            {ai_reply_structured}
        </div>
        """

    # Save conversation
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": structured_reply})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=2)

    return structured_reply

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

# --- Run Flask app ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
