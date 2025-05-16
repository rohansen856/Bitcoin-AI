import openai
import os
from dotenv import load_dotenv
load_dotenv()


class OpenAIEmbedder:
    def __init__(self):
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def embed(self, texts):
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        return [r["embedding"] for r in response["data"]]
