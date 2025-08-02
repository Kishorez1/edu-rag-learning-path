import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sentence_transformers import SentenceTransformer
import chromadb
import os
import json
import time

model = SentenceTransformer('all-MiniLM-L6-v2')
content_dir = "data/content/"
documents = []
metadatas = []
ids = []

for idx, filename in enumerate(os.listdir(content_dir)):
    with open(os.path.join(content_dir, filename), 'r') as f:
        text = f.read()
    documents.append(text)
    metadatas.append({"level": "beginner" if "basics" in filename else "intermediate", 
                        "style": "textual", 
                        "competency": filename.split('.')[0]})
    ids.append(f"doc{idx+1}")

embeddings = model.encode(documents)
client = chromadb.Client()
collection = client.create_collection("edu_content")
collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

query = "Learn Python basics"
query_embedding = model.encode(query)
where_filter = {"level": "beginner"} if "basics" in query.lower() else {}
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
    where=where_filter
  )

learning_path = sorted(
      zip(results['ids'][0], results['documents'][0], results['metadatas'][0]),
      key=lambda x: x[2]['level'] != 'beginner'
  )
print("Learning Path:")
for id, doc, meta in learning_path:
      print(f"Step: {meta['competency']} ({meta['level']}): {doc[:50]}...")

progress_entry = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "query": query,
    "learning_path": [meta for _, _, meta in learning_path]
  }
progress_file = "progress.json"
try:
    with open(progress_file, "r") as f:
        progress = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
      progress = []
progress.append(progress_entry)
with open(progress_file, "w") as f:
      json.dump(progress, f, indent=2)

print("\nUser Progress:")
for entry in progress:
      print(f"Query: {entry['query']} at {entry['timestamp']}")
      for meta in entry['learning_path']:
          print(f" - {meta['competency']} ({meta['level']})")

print("\nRaw Retrieval Results:")
for doc, meta, id in zip(results['documents'][0], results['metadatas'][0], results['ids'][0]):
      print(f"ID: {id}, Content: {doc[:50]}..., Metadata: {meta}")