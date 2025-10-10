from flask import Flask, render_template, request, jsonify
import json, os, requests
from dotenv import load_dotenv
import uuid
import re
from flask import send_file

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

# --- Persisted user name ---
USER_FILE = "user.json"
user_data = {}
if os.path.exists(USER_FILE):
    try:
        with open(USER_FILE, "r", encoding="utf-8") as f:
            user_data = json.load(f) or {}
    except Exception:
        user_data = {}

def set_user_name(name: str):
    global user_data
    if not name:
        return
    name = name.strip().capitalize()
    user_data["name"] = name
    try:
        with open(USER_FILE, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Could not save user name:", e)

def get_user_name():
    # Always reload from disk to ensure persistence across requests / restarts
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
                name = data.get("name")
                if name:
                    return name
        except Exception:
            pass
    # fallback: check memory-loaded user_data
    name = user_data.get("name")
    if name:
        return name
    # final fallback: scan chat history (rare)
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
    # always check persisted name
    user_name = get_user_name()

    # Step 1: If name not known, try to detect
    if not user_name:
        detected_name = store_name(user_input)
        if detected_name:
            user_name = detected_name
            # Persist the detected name so it's stored after the first time
            set_user_name(user_name)
            ai_reply = f"Nice to meet you, {user_name}! How are you feeling today?"
        else:
            ai_reply = "Hello! I’m Saathi, your mental health companion. May I know your name?"
    else:
        # include the stored name in the prompt so assistant can greet appropriately
        ai_reply = ask_perplexity(f"User name: {user_name}. {user_input}")

    ai_reply_structured = format_reply(ai_reply)

    # Save conversation (include name once if present)
    entry = {"role": "user", "content": user_input}
    if user_name:
        entry["name"] = user_name
    chat_history.append(entry)
    chat_history.append({"role": "assistant", "content": ai_reply})  # store raw assistant text
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, indent=2)
    except Exception as e:
        print("Could not save chat history:", e)

    # Return structured HTML for frontend
    structured_reply = f"""
    <div class="saathi-response">
        {ai_reply_structured}
    </div>
    """
    return structured_reply

# --- Planner storage ---
PLANNER_FILE = "planner.json"
if os.path.exists(PLANNER_FILE):
    with open(PLANNER_FILE, "r", encoding="utf-8") as f:
        try:
            planner_items = json.load(f)
        except Exception:
            planner_items = []
else:
    planner_items = []

def save_planner():
    with open(PLANNER_FILE, "w", encoding="utf-8") as f:
        json.dump(planner_items, f, ensure_ascii=False, indent=2)

# --- Planner API ---
@app.route("/planner_items", methods=["GET"])
def get_planner_items():
    return jsonify(planner_items)

@app.route("/planner_items", methods=["POST"])
def add_planner_item():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "Title required"}), 400
    item = {
        "id": str(uuid.uuid4()),
        "title": title,
        "date": data.get("date", "").strip(),
        "time": data.get("time", "").strip(),
        "notes": data.get("notes", "").strip(),
        "completed": False
    }
    planner_items.append(item)
    save_planner()
    return jsonify(item), 201

@app.route("/planner_items/<item_id>", methods=["DELETE"])
def delete_planner_item(item_id):
    global planner_items
    planner_items = [i for i in planner_items if i.get("id") != item_id]
    save_planner()
    return jsonify({"ok": True})

@app.route("/planner_items/<item_id>", methods=["PATCH"])
def update_planner_item(item_id):
    data = request.get_json(silent=True) or {}
    for it in planner_items:
        if it.get("id") == item_id:
            if "completed" in data:
                it["completed"] = bool(data["completed"])
            if "title" in data: it["title"] = (data.get("title") or "").strip()
            if "date" in data: it["date"] = (data.get("date") or "").strip()
            if "time" in data: it["time"] = (data.get("time") or "").strip()
            if "notes" in data: it["notes"] = (data.get("notes") or "").strip()
            save_planner()
            return jsonify(it)
    return jsonify({"error": "Not found"}), 404

# Download planner file (returns planner.json as attachment)
@app.route("/download_planner", methods=["GET"])
def download_planner():
    # return the same structured plain-text export so /download_planner does NOT send raw JSON
    return download_planner_text()

# new: download planner as structured plain text
@app.route("/download_planner_text", methods=["GET"])
def download_planner_text():
    # Build a readable plain-text export of planner_items
    lines = []
    if not planner_items:
        lines.append("Planner is empty.")
    else:
        for idx, it in enumerate(planner_items, start=1):
            lines.append(f"Item {idx}")
            lines.append(f"Title : {it.get('title','')}")
            lines.append(f"Date  : {it.get('date','')}")
            lines.append(f"Time  : {it.get('time','')}")
            notes = (it.get('notes') or "").strip()
            if notes:
                # preserve newlines in notes by indenting subsequent lines
                note_lines = notes.splitlines()
                lines.append(f"Notes : {note_lines[0]}")
                for nl in note_lines[1:]:
                    lines.append(f"        {nl}")
            else:
                lines.append("Notes : ")
            lines.append(f"Status: {'Completed' if it.get('completed') else 'Pending'}")
            lines.append("-" * 40)
    body = "\n".join(lines) + "\n"
    headers = {
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Disposition": 'attachment; filename="planner.txt"'
    }
    return (body, 200, headers)

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
