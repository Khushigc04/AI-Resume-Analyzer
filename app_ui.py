import streamlit as st
from app import load_uploaded_pdf, process_resume, analyze_resume

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("AI Resume Analyzer (RAG System)")
st.write("Upload your resume and get AI-powered insights.")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# -----------------------------
# PROCESS
# -----------------------------
if uploaded_file is not None:

    with st.spinner("Processing resume..."):

        try:
            # Load
            documents = load_uploaded_pdf(uploaded_file)

            # Create vector DB
            db = process_resume(documents)

            # Analyze
            result = analyze_resume(db)

            st.success("Analysis Complete!")
            st.subheader("AI Insights:")
            st.write(result)

        except Exception as e:
            st.error(f"Error: {str(e)}")