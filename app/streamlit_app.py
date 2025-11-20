# app/streamlit_app.py
"""
Expense Assistant - Streamlit app
Features:
 - Persistent chat & transactions (saved to app/user_data.json & app/chat_history.json)
 - Chat bubbles with timestamps
 - Model prediction + YAML keyword fallback on low confidence
 - Smart confirm dialog (when model low confidence)
 - Recent transactions table + bar chart summary
 - Clear / delete actions and "What can you do?" help
 - Emoji responses and larger input box
"""

import streamlit as st
import joblib
import json
import os
import yaml
import re
from datetime import datetime
from difflib import get_close_matches
import pandas as pd

# ---------------- CONFIG ----------------
MODEL_PATH = "models/transaction_model.joblib"
YAML_PATH = "categories.yaml"      # adjust if YAML is elsewhere
APP_DIR = "app"                    # folder to store JSON files
DATA_FILE = os.path.join(APP_DIR, "user_data.json")
CHAT_FILE = os.path.join(APP_DIR, "chat_history.json")

# thresholds
CONFIDENCE_THRESHOLD = 0.45        # below this, consult YAML fallback
SMART_CONFIRM_THRESHOLD = 0.35     # below this, ask user to confirm predicted category

# emoji map for categories (add more as needed)
EMOJI = {
    "Dining": "🍔",
    "Shopping": "🛍️",
    "Travel": "✈️",
    "Bills": "💡",
    "Entertainment": "🎬",
    "Groceries": "🛒",
    "Health": "🩺",
    "Housing": "🏠",
    "Subscriptions": "🔁",
    "Other": "🔖"
}

# ---------------- ENSURE APP FOLDER & FILES EXIST ----------------
os.makedirs(APP_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(CHAT_FILE):
    with open(CHAT_FILE, "w") as f:
        json.dump([], f)

# ---------------- HELPERS: load/save ----------------
def load_memory():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_memory(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

def load_chat():
    with open(CHAT_FILE, "r") as f:
        return json.load(f)

def save_chat(history):
    with open(CHAT_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)

# ---------------- LOAD MODEL ----------------
model = joblib.load(MODEL_PATH)
# pipeline steps names in your training code were "tfidf" and "clf"
vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["clf"]

# ---------------- YAML fallback with fuzzy matching ----------------
def load_yaml_rules():
    if not os.path.exists(YAML_PATH):
        return {}
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # flatten keywords: map category -> keywords (lowercase)
    rules = {}
    for cat, val in (data.get("categories") or {}).items():
        keys = val.get("keywords", []) if isinstance(val, dict) else []
        rules[cat] = [k.lower() for k in keys]
    return rules

YAML_RULES = load_yaml_rules()

def fallback_using_yaml(text):
    t = text.lower()
    # direct keyword match
    for category, keywords in YAML_RULES.items():
        for kw in keywords:
            if kw in t:
                return category
    # fuzzy match words against keywords (for misspellings)
    words = re.findall(r"[a-z0-9]+", t)
    for category, keywords in YAML_RULES.items():
        for w in words:
            match = get_close_matches(w, keywords, n=1, cutoff=0.8)
            if match:
                return category
    return None

# ---------------- PREDICTION & parsing ----------------
amount_re = re.compile(r"(\d+(?:\.\d{1,2})?)")  # find numeric amounts

def extract_amount(text):
    m = amount_re.search(text.replace(",", ""))
    # Lowercase for easier matching
    t = text.lower().replace(",", "")
    # Keywords to look for
    keywords = ["is", "for", "price", "amount", "cost", "at", "in"]
    # Regex: keyword (optional words) number
    for kw in keywords:
        # e.g. 'for 500', 'price is 1200', 'amount: 99.99'
        # Require a word boundary or space before the number
        pattern = rf"{kw}\s*(?:\w*\s*){{0,3}}(?:\b|\s)(\d+(?:\.\d{{1,2}})?)\b"
        match = re.search(pattern, t)
        if match:
            try:
                return float(match.group(1))
            except:
                continue
    # Fallback: largest number
    numbers = amount_re.findall(t)
    if numbers:
        try:
            return float(max(numbers, key=lambda x: float(x)))
        except:
            return None
    return None

def classify_text(text):
    X = vectorizer.transform([text])
    pred = classifier.predict(X)[0]
    probs = classifier.predict_proba(X)[0]
    confidence = float(max(probs))
    # also prepare top choices for smart confirm
    top_idx = probs.argsort()[::-1][:3]
    classes = list(classifier.classes_)
    top_choices = [(classes[i], float(probs[i])) for i in top_idx]

    # Explainability: get top features for the predicted class
    feature_names = vectorizer.get_feature_names_out()
    class_index = list(classifier.classes_).index(pred)
    # Get coefficients for the predicted class
    if hasattr(classifier, "coef_"):
        coefs = classifier.coef_[class_index]
        top_feat_idx = X.toarray()[0].argsort()[::-1]
        # Only consider features present in the input
        present_idx = [i for i in top_feat_idx if X.toarray()[0][i] > 0]
        top_words = [feature_names[i] for i in present_idx[:3]]
    else:
        top_words = []

    return pred, round(confidence, 3), top_choices, top_words

# ---------------- UI helpers ----------------
def add_chat(sender, message):
    history = load_chat()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append({"sender": sender, "message": message, "ts": ts})
    save_chat(history)

def render_bubble(item):
    sender = item["sender"]
    msg = item["message"]
    ts = item.get("ts", "")
    if sender == "user":
        st.markdown(f"""
        <div style="display:flex; margin:6px 0;">
          <div style="max-width:75%; margin-left:auto; background:#dcf8c6; padding:10px; border-radius:12px; text-align:right;">
            <div style="font-size:14px">{msg}</div>
            <div style="font-size:11px; color:#666; margin-top:6px;">{ts}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex; margin:6px 0;">
          <div style="max-width:75%; background:#f1f0f0; padding:10px; border-radius:12px;">
            <div style="font-size:14px">{msg}</div>
            <div style="font-size:11px; color:#666; margin-top:6px;">{ts}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Expense Assistant", layout="wide", page_icon="💰")
st.title("💬 Expense Assistant")
st.write("Talk to your personal finance buddy — type naturally (e.g. 'bought pizza for 250').")

# layout: left = chat, right = controls/summary
col1, col2 = st.columns([2, 1])

with col2:
    st.header("Quick Actions")
    if st.button("Clear All Data"):
        save_memory([])
        save_chat([])
        st.success("All transactions & chat cleared.")

    if st.button("Delete last transaction"):
        mem = load_memory()
        if mem:
            mem.pop()
            save_memory(mem)
            st.success("Last transaction deleted.")
        else:
            st.info("No transactions to delete.")

with col1:
    with st.expander("Show chat history", expanded=False):
        st.subheader("Chat")
        # show chat history (from saved file so it survives refresh)
        history = load_chat()
        for item in history[-200:]:
            render_bubble(item)

    st.markdown("### Speak to your assistant")
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""
    if st.session_state.get("clear_input", False):
        st.session_state["user_input"] = ""
        st.session_state["clear_input"] = False
    user_input = st.text_area("Type something...", height=80, key="user_input")

    # button: Send only
    send = st.button("Send")

    # Side-by-side: Recent transactions and This session summary with a good gap
    tcol1, spacer, tcol2 = st.columns([2, 0.3, 2])
    with tcol1:
        st.subheader("Recent transactions")
        mem = load_memory()
        if mem:
            df = pd.DataFrame(mem)
            if "amount" not in df.columns:
                df["amount"] = None
            # show last 10 with columns: text, category, date, amount (if available)
            show_df = df[["text", "category", "date", "amount"]].tail(10)
            show_df.columns = ["Text", "Category", "Date", "Amount"]
            # Convert Amount to string for Arrow compatibility
            show_df["Amount"] = show_df["Amount"].apply(lambda x: "" if pd.isna(x) else str(x))
            st.dataframe(show_df, height=240)
        else:
            st.write("No transactions yet.")
    with tcol2:
        st.subheader("This session summary")
        mem = load_memory()
        if mem:
            df = pd.DataFrame(mem)
            cat_sums = df["category"].value_counts().to_dict()
            for k, v in cat_sums.items():
                emoji = EMOJI.get(str(k), "")
                st.metric(f"{emoji} {k}", f"{v} txns")
        else:
            st.write("No data to summarize.")

    # send handling
    if send and user_input and user_input.strip():
        text = user_input.strip()
        # record user chat
        add_chat("user", text)

        # special commands
        if text.lower().strip() in ["summary", "show summary", "give summary"]:
            mem = load_memory()
            if not mem:
                bot_msg = "No transactions recorded yet."
                add_chat("bot", bot_msg)
            else:
                # prepare summary string
                df = pd.DataFrame(mem)
                counts = df["category"].value_counts().to_dict()
                s = "Here is your summary:\n\n"
                for k, v in counts.items():
                    s += f"• {k}: {v}\n"
                add_chat("bot", s)
            st.session_state["clear_input"] = True
            st.rerun()

        if text.lower().strip() in ["clear data", "reset", "reset data"]:
            save_memory([])
            save_chat([])
            add_chat("bot", "All data cleared.")
            st.session_state["clear_input"] = True
            st.rerun()

        # normal flow: predict
        pred, conf, top_choices, top_words = classify_text(text)

        # if model is very unsure, try YAML fallback
        chosen_category = pred
        used_fallback = False
        matched_keyword = None
        if conf < CONFIDENCE_THRESHOLD:
            fb = None
            # Modified fallback to also return matched keyword
            t = text.lower()
            for category, keywords in YAML_RULES.items():
                for kw in keywords:
                    if kw in t:
                        fb = category
                        matched_keyword = kw
                        break
                if fb:
                    break
            if not fb:
                words = re.findall(r"[a-z0-9]+", t)
                for category, keywords in YAML_RULES.items():
                    for w in words:
                        match = get_close_matches(w, keywords, n=1, cutoff=0.8)
                        if match:
                            fb = category
                            matched_keyword = match[0]
                            break
                    if fb:
                        break
            if fb:
                chosen_category = fb
                used_fallback = True

        # -------- FIX: ensure chosen_category is always a string --------
        if not chosen_category:
            chosen_category = "Other"

        # if still low confidence, show smart confirm box to user
        if conf < SMART_CONFIRM_THRESHOLD:
            # show options: top model choices + fallback if available + "Other"
            options = [c for c, _ in top_choices]
            if used_fallback and fb not in options:
                options = [fb] + options
            options = [o for o in options if o]  # remove empties
            # ensure uniqueness and preserve order
            seen = set()
            opts = []
            for o in options:
                if o not in seen:
                    opts.append(o); seen.add(o)
            opts.append("Other / None")
            st.markdown("**I'm unsure. Which category is correct?**")
            choice = st.radio("Choose category", opts, index=0)
            if st.button("Confirm category"):
                if choice != "Other / None":
                    chosen_category = choice
                # else keep chosen_category as-is (pred)
                # add to memory and chat below
        # extract amount if present
        amount = extract_amount(text)

        entry = {
            "text": text,
            "category": chosen_category,
            "date": datetime.now().isoformat(),
            "amount": amount if amount is not None else None
        }

        mem = load_memory()
        mem.append(entry)
        save_memory(mem)

        # prepare bot reply with explainability
        emoji = EMOJI.get(str(chosen_category), "")
        if used_fallback:
            if matched_keyword:
                bot_msg = f"{emoji} Added to **{chosen_category}** (via keyword: '{matched_keyword}')"
            else:
                bot_msg = f"{emoji} Added to **{chosen_category}** (via keywords)."
        else:
            if top_words:
                bot_msg = f"{emoji} Added to **{chosen_category}** (confidence: {conf})\nTop features: {', '.join(top_words)}"
            else:
                bot_msg = f"{emoji} Added to **{chosen_category}** (confidence: {conf})"
        add_chat("bot", bot_msg)

        # Set flag to clear input and rerun after all logic
        st.session_state["clear_input"] = True
        st.rerun()

# footer: small tips
st.markdown("---")
st.markdown("**Tips:** Try 'bought pizza dominos 350' or 'paid electricity 1200'.")
