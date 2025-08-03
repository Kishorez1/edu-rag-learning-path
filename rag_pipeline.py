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
import numpy as np

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
    level = "beginner" if "basics" in filename else "intermediate" if "intermediate" in filename else "advanced"
    metadatas.append({
        "level": level,
        "style": "textual",
        "competency": filename.split('.')[0]
    })
    ids.append(f"doc{idx+1}")

# Create ChromaDB collection
embeddings = model.encode(documents)
# Convert NumPy arrays to lists
embeddings = [emb.tolist() for emb in embeddings]
client = chromadb.Client()
collection = client.create_collection("edu_content")
collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

def preprocess_query(query):
    tokens = word_tokenize(query.lower())
    synonyms = {
        'basics': ['fundamentals', 'introduction', 'beginner'],
        'variables': ['data', 'types'],
        'functions': ['methods', 'procedures'],
        'classes': ['objects', 'oop'],
        'decorators': ['wrappers', 'modifiers'],
        'generators': ['iterators', 'yield'],
        'async': ['asynchronous', 'concurrent']
    }
    keywords = [token for token in tokens if token not in stop_words and token.isalnum()]
    expanded_keywords = []
    for token in keywords:
        expanded_keywords.append(token)
        for key, syn_list in synonyms.items():
            if token in syn_list:
                expanded_keywords.append(key)
    return list(set(expanded_keywords))

def process_query(query):
    # Query processing
    keywords = preprocess_query(query)
    where_filter = None
    if any(k in ['basics', 'variables', 'loops', 'fundamentals', 'introduction'] for k in keywords):
        where_filter = {"level": "beginner"}
    elif any(k in ['functions', 'classes', 'lists', 'methods', 'objects'] for k in keywords):
        where_filter = {"level": "intermediate"}
    elif any(k in ['decorators', 'generators', 'async', 'wrappers', 'iterators', 'concurrent'] for k in keywords):
        where_filter = {"level": "advanced"}

    query_embedding = model.encode(query)
    # Convert query embedding to list
    query_embedding = query_embedding.tolist()
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
        key=lambda x: {'beginner': 0, 'intermediate': 1, 'advanced': 2}[x[2]['level']]
    ) or sorted(zip(results['ids'][0], results['documents'][0], results['metadatas'][0]),
                key=lambda x: {'beginner': 0, 'intermediate': 1, 'advanced': 2}[x[2]['level']])

    # Generate LLM response
    path_summary = f"Based on your query '{query}', here's your personalized learning path:\n"
    for i, (_, _, meta) in enumerate(learning_path):
        path_summary += f"{i+1}. {meta['competency'].replace('_', ' ').title()} ({meta['level']} level)\n"
    llm_response = summarizer(path_summary, max_length=50, min_length=20, do_sample=False)[0]['summary_text']

    # Generate learning path details
    path_details = []
    for id, doc, meta in learning_path:
        status = "Completed" if meta['competency'] in completed_competencies else "To Learn"
        summary = summarizer(doc[:500], max_length=30, min_length=15, do_sample=False)[0]['summary_text']
        path_details.append({
            "id": id,
            "competency": meta['competency'],
            "level": meta['level'],
            "status": status,
            "summary": summary,
            "preview": doc[:50] + "..."
        })

    # Recommend uncompleted topics
    all_competencies = {meta['competency'] for meta in metadatas}
    recommendations = sorted(
        [comp for comp in all_competencies if comp not in completed_competencies],
        key=lambda x: {'python_basics': 0, 'python_intermediate': 1, 'python_advanced': 2}.get(x, 3)
    )

    # Save progress
    progress_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "keywords": keywords,
        "learning_path": [meta for _, _, meta in learning_path],
        "completed": learning_path[0][2]['competency'] if learning_path else None
    }
    progress.append(progress_entry)
    try:
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)
    except Exception:
        pass  # Skip saving in read-only environments

    return {
        "llm_response": llm_response,
        "path_details": path_details,
        "progress": progress,
        "recommendations": recommendations,
        "raw_results": [
            {"id": id, "content": doc[:50] + "...", "metadata": meta}
            for id, doc, meta in zip(results['ids'][0], results['documents'][0], results['metadatas'][0])
        ]
    }

if __name__ == "__main__":
    query = input("Enter your query (e.g., Learn Python basics): ")
    result = process_query(query)
    print("Learning Path Summary:")
    print(result['llm_response'])
    print("\nLearning Path Details:")
    for item in result['path_details']:
        print(f"Step: {item['competency']} ({item['level']}, {item['status']}):")
        print(f"Summary: {item['summary']}")
        print(f"Content Preview: {item['preview']}")
    print("\nUser Progress:")
    for entry in result['progress']:
        print(f"Query: {entry['query']} at {entry['timestamp']}, Keywords: {entry.get('keywords', [])}")
        for meta in entry['learning_path']:
            print(f" - {meta['competency']} ({meta['level']})")
    print("\nRecommended Topics:", result['recommendations'])
    print("\nRaw Retrieval Results:")
    for item in result['raw_results']:
        print(f"ID: {item['id']}, Content: {item['content']}, Metadata: {item['metadata']}")