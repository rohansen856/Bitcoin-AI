from vector_index.embed_models import get_embedder
from vector_index.db_clients import get_vector_db
import json
from pathlib import Path
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_index(docs: list, id : list, db_name: str, embedder_name: str, index_name : str):
    embedder = get_embedder(embedder_name)
    vector_db = get_vector_db(db_name, index_name, embedder.get_dimension())

    # docs = docs[:50]
    # id = id[:50]
    
    embeddings = embedder.embed([doc["body"] for doc in docs])
    vector_db.add_documents(id, embeddings, metadata=docs)
    return  vector_db, embedder

def load_and_chunk_json_files(data_dir: str) -> Tuple[List[dict], List[str]]:
    data_path = Path(data_dir)
    all_docs = []
    all_ids = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

    for file_path in data_path.rglob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Skipping {file_path} due to JSON error: {e}")
                continue

            for item in content:
                item_id = str(item.get("id") or item.get("_id") or "unknown")
                text = item.get("body") or item.get("body_text")

                if not text:
                    continue

                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{item_id}-{i}"
                    metadata = {k: v for k, v in item.items() if k not in ("id", "_id", "body", "body_text")}
                    metadata["body"] = chunk
                    all_docs.append(metadata)
                    all_ids.append(chunk_id)
                    
    print(len(all_ids))

    return all_docs, all_ids


if __name__ == "__main__":
    docs , id= load_and_chunk_json_files("./data")
    build_index(docs , id, "pinecone", "togetherai", "chatbtcrag")
    