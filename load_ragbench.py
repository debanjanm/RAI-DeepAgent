import json
import os
import glob

DATA_DIR = "open_ragbench_local"

def load_jsonl(filename):
    path = os.path.join(DATA_DIR, filename)
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

print("Loading dataset from local files...")

answers = load_jsonl("answers.jsonl")
print(f"Loaded {len(answers)} answers. Example: {answers[0]}")

queries = load_jsonl("queries.jsonl")
print(f"Loaded {len(queries)} queries. Example: {queries[0]}")

qrels = load_jsonl("qrels.jsonl")
print(f"Loaded {len(qrels)} qrels. Example: {qrels[0]}")

# Check corpus
corpus_files = glob.glob(os.path.join(DATA_DIR, "corpus", "*.json"))
print(f"Found {len(corpus_files)} corpus documents.")

if corpus_files:
    with open(corpus_files[0], 'r', encoding='utf-8') as f:
        doc = json.load(f)
    print(f"Example corpus document keys: {list(doc.keys())}")

print("\nDataset is ready to use from 'open_ragbench_local' directory.")

print("-" * 50)
print("Example using Hugging Face datasets library (if installed):")
print("""
from datasets import load_dataset

dataset = load_dataset('json', data_files={
    'answers': 'open_ragbench_local/answers.jsonl',
    'queries': 'open_ragbench_local/queries.jsonl',
    'qrels': 'open_ragbench_local/qrels.jsonl'
})

print(dataset)
""")
