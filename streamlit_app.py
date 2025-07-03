import os
# os.environ.setdefault("HF_HOME", "/home/user/huggingface_cache")
# os.environ.setdefault("TRANSFORMERS_CACHE", "/home/user/huggingface_cache/transformers")
# os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/home/user/huggingface_cache/sentence_transformers")
import streamlit as st
import openai
from collections import deque
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Setup (exact hardcoded keys you provided)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-bot")
model = SentenceTransformer('all-mpnet-base-v2')
chat_history = deque(maxlen=10)  # last 5 pairs = 10 messages

st.title("🔍 Legal RAG Assistant (Streamlit)")

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=10)

def get_rewritten_query(user_query):
    hist = list(st.session_state.history)[-4:]
    hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in hist)
    messages = [
        {"role": "system", "content":
         "You are a legal assistant that rewrites user queries into clear, context-aware queries for vector DB lookup. If its already clear then dont rewite"},
        {"role": "user", "content":
         f"History:\n{hist_text}\n\nNew query:\n{user_query}\n\n"
         "Rewrite if needed for clarity/search purposes. Otherwise, repeat exactly."}
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=100
        )
        rewritten = resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Rewrite error: {e}")
        rewritten = user_query
    st.session_state.history.append({"role": "assistant", "content": f"🔁 Rewritten query: {rewritten}"})
    return rewritten

def retrieve_documents(query, top_k=5):
    emb = model.encode(query).tolist()
    try:
        return index.query(vector=emb, top_k=top_k, include_metadata=True)['matches']
    except Exception as e:
        st.error(f"Retrieve error: {e}")
        return []

def generate_response(user_query, docs):
    context = "\n\n---\n\n".join(d['metadata']['text'] for d in docs)
    messages = [{"role": "system", "content":
                 "You are a helpful legal assistant. Use provided context from documents. Answer only using the context."}]
    messages.extend(st.session_state.history)
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"})
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Response error: {e}")
        reply = "Sorry, I encountered an error generating the answer."
    st.session_state.history.append({"role": "assistant", "content": reply})
    return reply

# Chat UI
with st.form("chat_input", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit = st.form_submit_button("Send")

if submit and user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    rewritten = get_rewritten_query(user_input)
    docs = retrieve_documents(rewritten)
    assistant_reply = generate_response(rewritten, docs)

# Display history
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")
