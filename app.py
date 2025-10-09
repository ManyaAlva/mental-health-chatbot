from flask import Flask, render_template, request, jsonify
import json, os, requests
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
PERPLEXITY_KEY = os.getenv("PERPLEXITY_API_KEY")

app = Flask(__name__)

# --- Chat history ---
HISTORY_FILE = "chat_history.json"
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        chat_history = json.load(f)
else:
    chat_history = []

# --- System prompt ---
system_prompt = """
You are Saathi, a friendly, caring, and empathetic mental health chatbot.
Always respond in a supportive, concise, and non-judgmental way.
Greet the user by their name if provided.
"""

# --- Ask AI / Perplexity ---
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

# --- Format AI reply to HTML ---
def format_reply(ai_text):
    import re

    def ensure_sentence_end(t: str) -> str:
        t = t.rstrip()
        if not t:
            return t
        if t[-1] not in ".!?…":
            return t + "."
        return t

    # Remove citation-style markers like [1], [1][3], etc.
    text = re.sub(r'(?:\s*\[\d+\])+', '', ai_text)

    # Convert bold **text** to <strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Remove any remaining asterisks (user requested no '*' shown)
    text = text.replace('*', '')

    # Ensure reply doesn't end abruptly
    text = ensure_sentence_end(text)

    lines = text.splitlines()
    output_parts = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        # list item if starts with -, * or numbered like "1."
        if re.match(r'^(-|\*)\s+', stripped) or re.match(r'^\d+\.\s+', stripped):
            if not in_list:
                in_list = True
                output_parts.append('<ul>')
            # remove leading marker and whitespace
            item = re.sub(r'^(-|\*|\d+\.)\s+', '', stripped)
            item = ensure_sentence_end(item)  # ensure each list item ends cleanly
            output_parts.append(f'<li>{item}</li>')
        else:
            if in_list:
                output_parts.append('</ul>')
                in_list = False
            if stripped:
                sentence = ensure_sentence_end(stripped)
                output_parts.append(f'<p>{sentence}</p>')

    if in_list:
        output_parts.append('</ul>')

    html = "\n".join(output_parts) if output_parts else f"<p>{ensure_sentence_end(text.strip())}</p>"
    return html

# --- Get stored user name ---
def get_user_name():
    for msg in chat_history:
        if msg.get("role") == "user" and "name" in msg:
            return msg["name"]
    return None

# --- Store name if detected ---
def store_name(user_input):
    stripped = user_input.strip()
    lower_input = stripped.lower()
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

    # If the entire input is just a greeting, ignore it
    if lower_input in greetings:
        return None

    # Detect "my name is", "i am", "i'm" followed by a name
    import re
    m = re.search(r"\b(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z'\-\s]*)", stripped, re.I)
    if m:
        name_part = m.group(1).strip()
        first = name_part.split()[0]
        return first.capitalize()

    # Single-word input treated as name
    if len(stripped.split()) == 1:
        return stripped.capitalize()

    return None

# --- Chatbot response ---
def chatbot_response(user_input):
    user_name = get_user_name()

    # Step 1: If name not known, try to detect
    if not user_name:
        detected_name = store_name(user_input)
        if detected_name:
            user_name = detected_name
            ai_reply = f"Nice to meet you, {user_name}! How are you feeling today?"
        else:
            ai_reply = "Hello! I’m Saathi, your mental health companion. May I know your name?"
    else:
        ai_reply = ask_perplexity(f"User name: {user_name}. {user_input}")

    ai_reply_structured = format_reply(ai_reply)

    # Step 2: Build final HTML
    structured_reply = f"""
    <div class="saathi-response">
        {ai_reply_structured}
    </div>
    """

    # Step 3: Save conversation with optional name
    entry = {"role": "user", "content": user_input}
    if user_name and not get_user_name():  # store name only once
        entry["name"] = user_name
    chat_history.append(entry)
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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
