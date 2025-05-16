def get_vector_db(name: str, index_name : str, dimension : int = 1024):
    if name == "pinecone":
        from .pinecone_client import PineconeClient
        return PineconeClient(index_name=index_name , dimension=dimension)
    else:
        raise ValueError(f"Unsupported vector DB: {name}")
