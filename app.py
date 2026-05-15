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

# ---------------- PREMIUM UI ---------------- #

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

/* Remove Streamlit Header */

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Main Hero */

.hero {
    text-align: center;
    padding-top: 20px;
    padding-bottom: 10px;
}

.hero-title {
    font-size: 70px;
    font-weight: 700;
    background: linear-gradient(to right, #60A5FA, #A855F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 20px;
    color: #CBD5E1;
    margin-bottom: 35px;
}

/* Glass Container */

.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(18px);
    border-radius: 24px;
    padding: 35px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 10px 40px rgba(0,0,0,0.35);
}

/* Input Box */

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: white;
    border-radius: 16px;
    padding: 18px;
    font-size: 17px;
}

/* Button */

.stButton button {
    width: 100%;
    border-radius: 16px;
    border: none;
    background: linear-gradient(90deg,#2563EB,#7C3AED);
    color: white;
    height: 58px;
    font-size: 18px;
    font-weight: 600;
    transition: 0.3s ease;
}

.stButton button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg,#1D4ED8,#6D28D9);
}

/* Answer Box */

.answer-box {
    background: rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 28px;
    margin-top: 25px;
    color: white;
    line-height: 1.9;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
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
    margin-bottom: 18px;
}

/* Cards */

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
    transform: translateY(-2px);
    background: rgba(255,255,255,0.1);
}

/* Footer */

.footer {
    text-align: center;
    margin-top: 50px;
    color: #94A3B8;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ---------------- #

st.markdown("""
<div class="hero">
    <div class="hero-title">🎓 SISTec AI Assistant</div>
    <div class="hero-subtitle">
        Premium AI Powered College Information Chatbot using Gemini + RAG
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

# ---------------- GEMINI ---------------- #

try:

    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    st.sidebar.success("✅ Gemini API Connected")

except Exception:

    st.error("❌ Gemini API Key not found")

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

# ---------------- MAIN CONTAINER ---------------- #

st.markdown('<div class="glass">', unsafe_allow_html=True)

question = st.text_input(
    "Ask anything about SISTec"
)

if st.button("✨ Generate Smart Answer"):

    if question.strip() == "":

        st.warning("Please enter a question")

    else:

        q_emb = model_emb.encode([question]).astype("float32")

        distances, indices = index.search(q_emb, k=4)

        context = "\n\n".join([texts[i] for i in indices[0]])

        prompt = f"""
Use only the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

        try:

            llm = genai.GenerativeModel("gemini-2.5-flash-lite")

            with st.spinner("🤖 AI is generating answer..."):

                response = llm.generate_content(prompt)

                st.markdown(
                    f"""
                    <div class="answer-box">
                    {response.text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with st.expander("📋 Context Used"):

                st.write(context)

        except Exception as e:

            st.error(f"Gemini Error: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #

st.markdown("""
<div class="footer">
Built with ❤️ using Python • Gemini API • FAISS • Streamlit
</div>
""", unsafe_allow_html=True)
