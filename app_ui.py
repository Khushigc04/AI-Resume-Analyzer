import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# Load API key
load_dotenv()

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("AI Resume Analyzer (RAG System)")
st.write("Upload your resume and get AI-powered insights.")

# Upload file
if uploaded_file is not None:
    documents = load_uploaded_pdf(uploaded_file)

if uploaded_file:

    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30
    )
    texts = splitter.split_documents(documents)

    # Embeddings + FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)

    # Input question
    query = st.text_input(
        "Ask a question about your resume",
        value="What skills are missing?"
    )

    if query:

        # Retrieve docs
        docs = db.similarity_search(query, k=5)

        # Filter context (clean logic)
        context_chunks = []

        for doc in docs:
            text = doc.page_content.lower()

            if (
                "technical skills" in text or
                "experience" in text or
                "project" in text
            ):
                context_chunks.append(doc.page_content)

        # Fallback
        if not context_chunks:
            context_chunks = [docs[0].page_content]

        # Remove duplicates
        context_chunks = list(set(context_chunks))
        context = "\n".join(context_chunks)

        # Groq client
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Final optimized prompt
        prompt = f"""
You are an expert career advisor.

Analyze ONLY the given resume content.

Identify:
- Missing technical skills
- Missing project experience

STRICT RULES:
- Give ONLY 3 points
- Each point must be specific to the resume
- No generic suggestions

Resume:
{context}

Question: {query}
"""

        # LLM call
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )

        # UI Output
        st.success("Analysis Complete ✅")
        st.subheader("AI Response:")
        st.markdown(response.choices[0].message.content)