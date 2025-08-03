Educational Content RAG with Learning Path Generation
  A Retrieval-Augmented Generation (RAG) system for personalized Python learning paths, built with Streamlit for interactive use.
Project Overview
  This project implements a RAG system to:

Retrieve relevant Python educational content based on user queries.
Generate personalized learning paths with beginner, intermediate, and advanced topics.
Summarize content using the t5-small LLM.
Track user progress in progress.json.
Recommend uncompleted topics.
Provide a Streamlit UI for user interaction.

Setup Instructions

Clone the Repository:
git clone https://github.com/Kishorez1/edu-rag-learning-path.git
cd edu-rag-learning-path


Install Rust (required for tokenizers):
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env


Create Virtual Environment:
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Run Locally:
streamlit run app.py


Access at http://localhost:8501.
Enter queries like "Learn Python basics" or "Python decorators".



File Structure

app.py: Streamlit UI for interactive queries.
rag_pipeline.py: Core RAG pipeline with query processing, learning path generation, and LLM integration.
fix_progress.py: Script to update progress.json with keywords.
requirements.txt: Project dependencies.
progress.json: Tracks user queries and progress.
data/content/: Contains python_basics.txt, python_intermediate.txt, python_advanced.txt.
.streamlit/config.toml: Streamlit deployment settings.

Deployment

Deployed on Streamlit Cloud: App URL (replace with your app URL).
Follow Streamlit Cloud to deploy from the main branch.

Usage

Enter a query in the Streamlit UI (e.g., "Learn Python functions").
View the learning path, summaries, progress, and recommended topics.
Progress is saved in progress.json locally (may be read-only on Streamlit Cloud).

Technologies

Python 3.9
Streamlit 1.39.0
Sentence-Transformers 2.2.2
ChromaDB 0.4.24
NLTK 3.8.1
Transformers 4.44.2 (with t5-small)
Huggingface Hub 0.23.2
NumPy 1.26.4

Submission
  Prepared for submission by August 4, 2025. Repository: https://github.com/Kishorez1/edu-rag-learning-path.