import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader

st.set_page_config(page_title="SISTec AI Assistant")

st.title("🎓 SISTec AI Assistant")

st.sidebar.header("Sample Questions")

st.sidebar.write("What courses are available?")
st.sidebar.write("Tell me about placements")
st.sidebar.write("Does SISTec provide hostel?")
st.sidebar.write("Where is SISTec located?")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.sidebar.success("✅ Gemini API Connected")

except Exception:
    st.error("❌ Gemini API Key not found")
    st.stop()

pdf_path = "sistec_info.pdf"

with st.spinner("Reading SISTec information..."):

    reader = PdfReader(pdf_path)

    full_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            full_text += text + "\n"

def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks

texts = chunk_text(full_text)

with st.spinner("Creating AI search index..."):

    model_emb = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model_emb.encode(texts).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)

st.success(f"✅ SISTec knowledge loaded successfully!")

question = st.text_input("Ask anything about SISTec")

if st.button("Get Answer"):

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

            with st.spinner("AI is thinking..."):

                response = llm.generate_content(prompt)

                st.subheader("Answer")

                st.write(response.text)

            with st.expander("📋 Context Used"):
                st.write(context)

        except Exception as e:
            st.error(f"Gemini Error: {e}")
