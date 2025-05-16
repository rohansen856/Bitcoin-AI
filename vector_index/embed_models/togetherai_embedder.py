from together import Together
import numpy as np
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()


class TogetherAIEmbedder:
    def __init__(self):
        self.api_key = os.environ.get("TOGETHER-API-KEY")
        self.client = Together(api_key = self.api_key)
        self.model = "BAAI/bge-large-en-v1.5"

    def embed(self, input_texts: List[str]) -> np.ndarray:
        """Generate embeddings from Together python library.

        Args:
            input_texts: a list of string input texts.
            model_api_string: str. An API string for a specific embedding model of your choice.

        Returns:
            embeddings_list: a list of embeddings. Each element corresponds to the each input text.
        """
        
        
        
        outputs = self.client.embeddings.create(
            input=input_texts,
            model=self.model,
        )
        return np.array([x.embedding for x in outputs.data])
    
    def get_dimension(self):
        return self.embed(["hello"])[0].shape[0]
