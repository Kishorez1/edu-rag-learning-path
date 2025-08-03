import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sentence_transformers import SentenceTransformer
import chromadb
import os
import json
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="t5-small")

# Load content
content_dir = "data/content/"
documents = []
metadatas = []
ids = []

for idx, filename in enumerate(os.listdir(content_dir)):
    with open(os.path.join(content_dir, filename), 'r') as f:
        text = f.read()
    documents.append(text)
    metadatas.append({
        "level": "beginner" if "basics" in filename else "intermediate",
        "style": "textual",
        "competency": filename.split('.')[0]
    })
    ids.append(f"doc{idx+1}")

# Create ChromaDB collection
embeddings = model.encode(documents)
client = chromadb.Client()
collection = client.create_collection("edu_content")
collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

def preprocess_query(query):
    tokens = word_tokenize(query.lower())
    synonyms = {
        'basics': ['fundamentals', 'introduction', 'beginner'],
        'variables': ['data', 'types'],
        'functions': ['methods', 'procedures'],
        'classes': ['objects', 'oop']
    }
    keywords = [token for token in tokens if token not in stop_words and token.isalnum()]
    expanded_keywords = []
    for token in keywords:
        expanded_keywords.append(token)
        for key, syn_list in synonyms.items():
            if token in syn_list:
                expanded_keywords.append(key)
    return list(set(expanded_keywords))

# Query processing
query = "Python Loops"
keywords = preprocess_query(query)
where_filter = None
if any(k in ['basics', 'variables', 'loops', 'fundamentals', 'introduction'] for k in keywords):
    where_filter = {"level": "beginner"}
elif any(k in ['functions', 'classes', 'lists', 'methods', 'objects'] for k in keywords):
    where_filter = {"level": "intermediate"}

query_embedding = model.encode(query)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    where=where_filter
)

# Load progress
progress_file = "progress.json"
try:
    with open(progress_file, "r") as f:
        progress = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    progress = []

# Track completed competencies
completed_competencies = set()
for entry in progress:
    completed = entry.get('completed')
    if completed:
        completed_competencies.add(completed)

# Generate learning path
learning_path = sorted(
    [(id, doc, meta) for id, doc, meta in zip(results['ids'][0], results['documents'][0], results['metadatas'][0])
     if meta['competency'] not in completed_competencies],
    key=lambda x: x[2]['level'] != 'beginner'
) or sorted(zip(results['ids'][0], results['documents'][0], results['metadatas'][0]),
            key=lambda x: x[2]['level'] != 'beginner')

# Generate LLM response
path_summary = f"Based on your query '{query}', here's your personalized learning path:\n"
for i, (_, _, meta) in enumerate(learning_path):
    path_summary += f"{i+1}. {meta['competency'].replace('_', ' ').title()} ({meta['level']} level)\n"

llm_response = summarizer(path_summary, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
print("Learning Path Summary:")
print(llm_response)

# Print learning path with summaries
print("\nLearning Path Details:")
for id, doc, meta in learning_path:
    status = "Completed" if meta['competency'] in completed_competencies else "To Learn"
    summary = summarizer(doc[:500], max_length=30, min_length=15, do_sample=False)[0]['summary_text']
    print(f"Step: {meta['competency']} ({meta['level']}, {status}):")
    print(f"Summary: {summary}")
    print(f"Content Preview: {doc[:50]}...")

# Save progress
progress_entry = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "query": query,
    "keywords": keywords,
    "learning_path": [meta for _, _, meta in learning_path],
    "completed": learning_path[0][2]['competency'] if learning_path else None
}
progress.append(progress_entry)
with open(progress_file, "w") as f:
    json.dump(progress, f, indent=2)

# Print progress
print("\nUser Progress:")
for entry in progress:
    print(f"Query: {entry['query']} at {entry['timestamp']}, Keywords: {entry.get('keywords', [])}")
    for meta in entry['learning_path']:
        print(f" - {meta['competency']} ({meta['level']})")

print("\nRaw Retrieval Results:")
for doc, meta, id in zip(results['documents'][0], results['metadatas'][0], results['ids'][0]):
    print(f"ID: {id}, Content: {doc[:50]}..., Metadata: {meta}")