import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="SISTec AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# ---------------- PREMIUM CSS ---------------- #

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background */

.stApp {
    background: linear-gradient(135deg, #020617, #0F172A, #1E1B4B);
    color: white;
}

/* Hide Streamlit Branding */

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Hero Section */

.hero {
    text-align: center;
    padding-top: 10px;
    padding-bottom: 25px;
}

.hero-title {
    font-size: 72px;
    font-weight: 700;
    background: linear-gradient(to right, #60A5FA, #A855F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 20px;
    color: #CBD5E1;
    margin-top: 10px;
}

/* Feature Cards */

.feature-container {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 35px;
    flex-wrap: wrap;
}

.feature-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px 28px;
    color: white;
    min-width: 220px;
    text-align: center;
    backdrop-filter: blur(10px);
    font-size: 17px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.98);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Sidebar Head */

.side-head {
    font-size: 24px;
    font-weight: 600;
    color: white;
    margin-bottom: 20px;
}

/* Sidebar Cards */

.card {
    background: rgba(255,255,255,0.07);
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 14px;
    color: #E2E8F0;
    border: 1px solid rgba(255,255,255,0.08);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-3px);
    background: rgba(255,255,255,0.1);
}

/* Input */

.stTextInput {
    max-width: 950px;
    margin: auto;
}

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    color: white;
    border-radius: 20px;
    padding: 18px;
    font-size: 18px;
}

/* Button */

.stButton button {
    width: 100%;
    border-radius: 18px;
    border: none;
    background: linear-gradient(90deg,#2563EB,#7C3AED);
    color: white;
    height: 60px;
    font-size: 18px;
    font-weight: 600;
    transition: 0.3s ease;
}

.stButton button:hover {
    transform: scale(1.03);
}

/* Answer Box */

.answer-box {
    background: rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 30px;
    margin-top: 30px;
    color: white;
    line-height: 2;
    font-size: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}

/* Footer */

.footer {
    text-align: center;
    margin-top: 60px;
    color: #94A3B8;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ---------------- #

st.markdown("""
<div class="hero">
    <div class="hero-title">🎓 SISTec AI Assistant</div>
    <div class="hero-subtitle">
        Premium AI Powered College Information Chatbot using Gemini + RAG
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE CARDS ---------------- #

st.markdown("""
<div class="feature-container">

    <div class="feature-card">
        🤖 AI Powered Assistant
    </div>

    <div class="feature-card">
        ⚡ Instant Smart Answers
    </div>

    <div class="feature-card">
        📚 PDF Based Knowledge
    </div>

</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.markdown(
        '<div class="side-head">📌 Sample Questions</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card">What courses are available in SISTec?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card">Tell me about placements</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card">Does SISTec provide hostel facility?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card">Where is SISTec located?</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.success("⚡ Smart Assistant Ready")

# ---------------- GEMINI API ---------------- #

try:

    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

except Exception:

    st.error("Gemini API Key not found")

    st.stop()

# ---------------- LOAD PDF ---------------- #

pdf_path = "sistec_info.pdf"

with st.spinner("📚 Loading SISTec knowledge base..."):

    reader = PdfReader(pdf_path)

    full_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            full_text += text + "\n"

# ---------------- CHUNKING ---------------- #

def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks

texts = chunk_text(full_text)

# ---------------- EMBEDDINGS ---------------- #

with st.spinner("⚡ Creating AI vector search engine..."):

    model_emb = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model_emb.encode(texts).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)

# ---------------- QUESTION INPUT ---------------- #

question = st.text_input(
    "Ask anything about SISTec"
)

# ---------------- BUTTON ---------------- #

col1, col2, col3 = st.columns([1,1,1])

with col2:

    generate = st.button("✨ Generate Smart Answer")

# ---------------- ANSWER ---------------- #

if generate:

    if question.strip() == "":

        st.warning("Please enter a question")

    else:

        q_emb = model_emb.encode([question]).astype("float32")

        distances, indices = index.search(q_emb, k=4)

        context = "\n\n".join([texts[i] for i in indices[0]])

        prompt = f"""
You are a SISTec college assistant.

Answer ONLY from the provided context.

If the answer is not available in the context, reply:
'Sorry, I can only answer questions related to SISTec.'

Context:
{context}

Question:
{question}

Answer:
"""

        try:

            llm = genai.GenerativeModel("gemini-2.5-flash-lite")

            with st.spinner("🤖 AI is analyzing SISTec knowledge..."):

                response = llm.generate_content(prompt)

                st.markdown(
                    f'''
                    <div class="answer-box">
                    {response.text}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            with st.expander("📋 View Context Used"):

                st.write(context)

        except Exception as e:

            st.error(f"Gemini Error: {e}")

# ---------------- FOOTER ---------------- #

st.markdown("""
<div class="footer">
Powered by Gemini AI • FAISS • Streamlit
</div>
""", unsafe_allow_html=True)
