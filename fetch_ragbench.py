import os
import json
import shutil
from pathlib import Path

# Try to import huggingface_hub, provide instructions if missing
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: 'huggingface_hub' library is required.")
    print("Please install it by running: pip install huggingface_hub")
    exit(1)

REPO_ID = "vectara/open_ragbench"
LOCAL_DIR = "open_ragbench_local"
TEMP_DIR = "temp_ragbench_download"
# The dataset files are located in this subfolder within the repo
REPO_SUBFOLDER = "pdf/arxiv" 

def convert_to_jsonl(source_path, dest_path, key_name, value_name=None):
    """
    Converts a JSON dictionary to a JSONL file (Line-delimited JSON).
    """
    print(f"Processing {source_path.name} -> {dest_path.name}...")
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(dest_path, 'w', encoding='utf-8') as f:
            if isinstance(data, dict):
                for k, v in data.items():
                    record = {key_name: k}
                    if value_name:
                        # If the value is a direct string (like in answers.json)
                        record[value_name] = v
                    elif isinstance(v, dict):
                        # If the value is a dictionary (like in queries.json), merge it
                        record.update(v)
                    else:
                        # Fallback
                        record['content'] = v
                    f.write(json.dumps(record) + '\n')
            else:
                print(f"Warning: {source_path.name} is not a dictionary. Copying as is.")
                # If it's not a dict, maybe it's already a list? 
                # For this specific dataset, we know they are dicts, but good to be safe.
                pass
    except Exception as e:
        print(f"Failed to convert {source_path.name}: {e}")

def main():
    print(f"Downloading dataset '{REPO_ID}'...")
    print("This may take a while depending on your internet connection...")
    
    # Download specific files from the repository
    # We download to a temporary directory first
    download_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=f"{REPO_SUBFOLDER}/*",
        local_dir=TEMP_DIR,
        local_dir_use_symlinks=False  # Ensure we get actual files
    )
    
    # The path where the interesting files are
    source_base = Path(download_path) / REPO_SUBFOLDER
    dest_base = Path(LOCAL_DIR)
    
    # Create destination directory
    if not dest_base.exists():
        os.makedirs(dest_base)
        
    print(f"\nPreparing local dataset in '{LOCAL_DIR}'...")

    # 1. Convert answers.json -> answers.jsonl
    # Structure: { "uuid": "answer_text" }
    convert_to_jsonl(
        source_base / "answers.json", 
        dest_base / "answers.jsonl", 
        key_name="query_id", 
        value_name="answer"
    )

    # 2. Convert queries.json -> queries.jsonl
    # Structure: { "uuid": { "query": "...", "type": "...", ... } }
    convert_to_jsonl(
        source_base / "queries.json", 
        dest_base / "queries.jsonl", 
        key_name="query_id"
    )

    # 3. Convert qrels.json -> qrels.jsonl
    # Structure: { "uuid": { "doc_id": "...", "section_id": ... } }
    convert_to_jsonl(
        source_base / "qrels.json", 
        dest_base / "qrels.jsonl", 
        key_name="query_id"
    )

    # 4. Copy pdf_urls.json (keep as json)
    print("Copying pdf_urls.json...")
    shutil.copy(source_base / "pdf_urls.json", dest_base / "pdf_urls.json")

    # 5. Copy corpus directory
    source_corpus = source_base / "corpus"
    dest_corpus = dest_base / "corpus"
    
    if source_corpus.exists():
        print("Copying corpus directory (this contains the parsed PDFs)...")
        if dest_corpus.exists():
            shutil.rmtree(dest_corpus)
        shutil.copytree(source_corpus, dest_corpus)
    else:
        print("Warning: Corpus directory not found in download.")

    # Cleanup temporary download
    print("Cleaning up temporary files...")
    shutil.rmtree(TEMP_DIR)

    print("\nDone! Dataset is ready.")
    print(f"Location: {os.path.abspath(LOCAL_DIR)}")

if __name__ == "__main__":
    main()
