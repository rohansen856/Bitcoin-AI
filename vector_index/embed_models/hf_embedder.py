from sentence_transformers import SentenceTransformer
import torch
import os
from dotenv import load_dotenv
load_dotenv()


class HuggingFaceEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True).cuda()

    def embed(self, texts):
        query_embeddings = self.model.encode(texts)
        return query_embeddings.numpy().tolist()
    
    def get_dimension(self):
        return self.model.encode(["hello"]).numpy()[0].shape[0]


