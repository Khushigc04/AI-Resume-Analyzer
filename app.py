import os
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
# STEP 1: LOAD RESUME
# -----------------------------
loader = PyPDFLoader(tmp_path)
documents = loader.load()

print("Resume loaded successfully!")

# -----------------------------
# STEP 2: SPLIT TEXT (IMPROVED)
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,        # increased
    chunk_overlap=50       # increased
)

texts = splitter.split_documents(documents)

print("Text split into chunks:", len(texts))

# -----------------------------
# STEP 3: EMBEDDINGS
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.from_documents(texts, embeddings)

print("Embeddings created and stored!")

# -----------------------------
# STEP 4: RETRIEVAL (IMPROVED)
# -----------------------------
query = """
Find all technical sections including:
- programming languages
- frameworks
- tools
- project descriptions
- technologies used
- work experience technical details
"""

docs = db.similarity_search(query, k=10)

# FULL CONTEXT (NO BAD FILTERING)
context = "\n\n".join([doc.page_content for doc in docs])

print("\n--- RETRIEVED CONTEXT ---\n")
print(context[:500])
print("\nContext length:", len(context))

# -----------------------------
# STEP 5: LLM (FIXED MODEL)
# -----------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

prompt = f"""
You are a strict technical recruiter.

Rules:
- Use ONLY the provided resume content
- Do NOT hallucinate or invent any experience
- Avoid generic suggestions (like cloud, agile) unless clearly justified
- Focus on practical, role-relevant technical gaps

Resume Content:
{context}

Task:
Identify 3-4 meaningful missing technical gaps.

Each point must:
- Be based on missing evidence in the resume
- Be specific and relevant to software development roles
- NOT be generic advice that applies to everyone

Important:
Prefer concrete gaps like:
- missing frameworks
- missing tools
- missing project experience
- missing real-world implementation

Avoid:
- vague or universal suggestions (cloud, agile, etc.) unless strongly justified

Output format:
1. Missing Skill → Why it is missing (based on resume)
"""
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    temperature=0,
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\n--- FINAL ANSWER ---\n")
answer = response.choices[0].message.content

print("\n--- FINAL ANSWER ---\n")

validated_points = []

for line in answer.split("\n"):
    if "→" in line:
        line_lower = line.lower()

        # reject if contains tools NOT in context
        hallucination_words = [
            "selenium", "protractor", "docker", "kubernetes",
            "react", "angular", "nosql", "mongodb", "cypress",
            "jest"
        ]

        hallucinated = False
        for word in hallucination_words:
            if word in line_lower and word not in context.lower():
                hallucinated = True
                break

        if not hallucinated:
            validated_points.append(line)

# FINAL OUTPUT
if len(validated_points) >= 2:
    print("\n".join(validated_points))
else:
    print("⚠️ Output not reliable (hallucination detected)")
    print(answer)