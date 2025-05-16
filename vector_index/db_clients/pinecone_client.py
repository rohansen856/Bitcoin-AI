from pinecone import Pinecone, ServerlessSpec
import os
from typing import List, Dict, Optional
from uuid import uuid4

class PineconeClient:
    def __init__( self, api_key: Optional[str] = None,
                 index_name: str = "default-index", 
                 dimension: int = 1536, 
                 metric: str = "cosine"):
        
        self.api_key = api_key or os.getenv("PINECONE-API-KEY")
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        pc = Pinecone( api_key=self.api_key)

        if not pc.has_index(index_name):
            pc.create_index(
                name = index_name, 
                dimension = dimension, 
                metric = metric,
                spec = ServerlessSpec(cloud='aws', region = "us-east-1"),
                tags={ "environment": "development"}
            )

        self.index = pc.Index(name = index_name)
        print("Index Created Successfully")

    def add_documents(self, id : List[str], embeddings: List[List[float]], metadata: List[Dict] = None):
        
        
        vectors = [
            {"id": _id , "values": vec, "metadata": meta}
            for _id,  vec, meta in zip(id, embeddings, metadata)
        ]
        
        self.index.upsert(vectors=vectors, namespace = "")
        print("Documents added successfully")

    def exists(self, id: str) -> bool:
        response = self.index.fetch(ids=[id])
        return id in response.get("vectors", {})

    def query(self, vector: List[float], top_k: int = 5):
        return self.index.query(namespace="", vector=vector, top_k=top_k, include_metadata=True)
