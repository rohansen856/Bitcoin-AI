def get_embedder(name: str):
    if name == "huggingface":
        from .hf_embedder import HuggingFaceEmbedder
        return HuggingFaceEmbedder()
    elif name == "openai":
        from .openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder()
    elif name == "togetherai":
        from .togetherai_embedder import TogetherAIEmbedder
        return TogetherAIEmbedder()
    else:
        raise ValueError(f"Unsupported embedder: {name}")
