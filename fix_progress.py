import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

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

     # Load progress
with open("progress.json", "r") as f:
         progress = json.load(f)

     # Update entries
for entry in progress:
         if not entry.get('keywords'):
             entry['keywords'] = preprocess_query(entry['query'])

     # Save updated progress
with open("progress.json", "w") as f:
         json.dump(progress, f, indent=2)