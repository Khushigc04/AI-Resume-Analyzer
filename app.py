import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

# -----------------------------
# LOAD PDF FROM UPLOAD
# -----------------------------
def load_uploaded_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    return loader.load()

# -----------------------------
# PROCESS DOCUMENTS (RAG)
# -----------------------------
def process_resume(documents):

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    texts = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(texts, embeddings)

    return db


# -----------------------------
# ANALYZE WITH GROQ
# -----------------------------
def analyze_resume(db):

    query = "technical skills projects experience technologies"

    docs = db.similarity_search(query, k=5)

    context = "\n".join([doc.page_content for doc in docs])

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
You are an expert career advisor.

Analyze ONLY the given resume content.

STRICT RULES:
- Do NOT assume anything not present
- Do NOT hallucinate
- If data missing → say "Not mentioned"

Resume:
{context}

Question:
What important skills or experience are missing?

Give 3–4 bullet points.
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",   # working model
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content