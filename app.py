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

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #050816, #0b1026, #16113a);
    color: white;
}

section[data-testid="stSidebar"] {
    background: #0b1225;
    border-right: 1px solid rgba(255,255,255,0.08);
}

.main-title {
    text-align: center;
    font-size: 70px;
    font-weight: 700;
    background: linear-gradient(to right, #7aa2ff, #b06cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 10px;
}

.sub-title {
    text-align: center;
    color: #c7c9d3;
    font-size: 22px;
    margin-bottom: 40px;
}

.feature-container {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 40px;
    flex-wrap: wrap;
}

.feature-card {
    background: rgba(255,255,255,0.06);
    padding: 18px 28px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 18px;
    color: white;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0,0,0,0.2);
}

.answer-box {
    background: rgba(255,255,255,0.06);
    padding: 30px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-top: 25px;
    color: white;
    font-size: 18px;
    line-height: 1.8;
}

.footer {
    text-align: center;
    margin-top: 60px;
    color: #a8acc4;
    font-size: 15px;
}

.stTextInput > div > div > input {
    background-color: rgba(255,255,255,0.08);
    color: white;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.1);
    height: 55px;
    font-size: 18px;
}

.stButton > button {
    background: linear-gradient(to right, #3b82f6, #9333ea);
    color: white;
    border: none;
    border-radius: 16px;
    height: 55px;
    width: 260px;
    font-size: 18px;
    font-weight: 600;
}

.stButton > button:hover {
    transform: scale(1.03);
    transition: 0.3s;
}

.sample-box {
    background: rgba(255,255,255,0.06);
    padding: 16px;
    border-radius: 16px;
    margin-bottom: 14px;
    color: white;
    border: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

st.sidebar.markdown("## 📌 Sample Questions")

st.sidebar.markdown('<div class="sample-box">What courses are available in SISTec?</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sample-box">Tell me about placements</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sample-box">Does SISTec provide hostel facility?</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sample-box">Where is SISTec located?</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.success("⚡ Smart Assistant Ready")

# ---------------- GEMINI API ---------------- #

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.error("API Key Missing")
    st.stop()

# ---------------- TITLE ---------------- #

st.markdown('<div class="main-title">🎓 SISTec AI Assistant</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="sub-title">Premium AI Powered College Information Chatbot using Gemini + RAG</div>',
    unsafe_allow_html=True
)

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

# ---------------- PDF READING ---------------- #

pdf_path = "sistec_info.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# ---------------- TEXT CHUNKING ---------------- #

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

model_emb = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model_emb.encode(texts).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])

index.add(embeddings)

# ---------------- USER QUESTION ---------------- #

question = st.text_input("Ask anything about SISTec")

# ---------------- BUTTON ---------------- #

if st.button("✨ Generate Smart Answer"):

    if question.strip() == "":

        st.warning("Please enter a question")

    else:

        q_emb = model_emb.encode([question]).astype("float32")

        distances, indices = index.search(q_emb, k=4)

        context = "\n\n".join([texts[i] for i in indices[0]])

        prompt = f"""
You are a professional AI college assistant.

Answer ONLY from the provided context.

If the answer is not available in context, say:
"Sorry, I could not find this information in the SISTec database."

Context:
{context}

Question:
{question}

Answer:
"""

        try:

            llm = genai.GenerativeModel("gemini-1.5-flash")

            response = llm.generate_content(prompt)

            st.markdown(
                f'<div class="answer-box">{response.text}</div>',
                unsafe_allow_html=True
            )

            with st.expander("📄 Context Used"):
                st.write(context)

        except Exception as e:

            st.error(f"Gemini Error: {e}")

# ---------------- FOOTER ---------------- #

st.markdown(
    '<div class="footer">Built with ❤️ using Python • Gemini API • FAISS • Streamlit</div>',
    unsafe_allow_html=True
)
