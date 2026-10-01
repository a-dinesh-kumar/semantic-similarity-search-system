from pathlib import Path

from fastapi import FastAPI, Query
from gensim.models import Word2Vec, FastText

from app.data_loader import load_knowledge_base
from app.search import (build_document_vectors, semantic_search)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# Model paths
# --------------------------------------------------

CBOW_MODEL_PATH = MODEL_DIR / "word2vec_cbow.model"
SKIPGRAM_MODEL_PATH = MODEL_DIR / "word2vec_skipgram.model"
FASTTEXT_MODEL_PATH = MODEL_DIR / "fasttext.model"


# --------------------------------------------------
# Load embedding models
# --------------------------------------------------

print("Loading embedding models...")

word2vec_cbow = Word2Vec.load(
    str(CBOW_MODEL_PATH),
    mmap="r"
)

print("✓ Word2Vec CBOW loaded")


word2vec_skipgram = Word2Vec.load(
    str(SKIPGRAM_MODEL_PATH),
    mmap="r"
)

print("✓ Word2Vec Skip-Gram loaded")


fasttext_model = FastText.load(
    str(FASTTEXT_MODEL_PATH),
    mmap="r"
)

print("✓ FastText loaded")


print("All embedding models loaded successfully.")


# --------------------------------------------------
# Model Registry
# --------------------------------------------------

MODELS = {
    "word2vec_cbow": word2vec_cbow,
    "word2vec_skipgram": word2vec_skipgram,
    "fasttext": fasttext_model
}


# --------------------------------------------------
# Load knowledge base
# --------------------------------------------------

knowledge_base = load_knowledge_base()


# --------------------------------------------------
# Build document vectors
# --------------------------------------------------

# print("Building document vectors...")

# document_vectors_cbow = build_document_vectors(
#     knowledge_base["cleaned_text"],
#     word2vec_cbow
# )

# print(
#     f"✓ CBOW document vectors created: "
#     f"{document_vectors_cbow.shape}"
# )


# document_vectors_skipgram = build_document_vectors(
#     knowledge_base["cleaned_text"],
#     word2vec_skipgram
# )

# print(
#     f"✓ Skip-Gram document vectors created: "
#     f"{document_vectors_skipgram.shape}"
# )


# document_vectors_fasttext = build_document_vectors(
#     knowledge_base["cleaned_text"],
#     fasttext_model
# )

# print(
#     f"✓ FastText document vectors created: "
#     f"{document_vectors_fasttext.shape}"
# )


# print("All document vectors created successfully.")


print("Building Skip-Gram document vectors...")

document_vectors_skipgram = build_document_vectors(
    knowledge_base["cleaned_text"],
    word2vec_skipgram
)

print(
    f"✓ Skip-Gram document vectors created: "
    f"{document_vectors_skipgram.shape}"
)

print("Document vectorization completed.")


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Semantic Similarity Search API",
    description="Semantic search using Word2Vec and FastText embeddings.",
    version="1.0.0"
)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Semantic Similarity Search API is running",
        "status": "success"
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models": {
            "word2vec_cbow": "loaded",
            "word2vec_skipgram": "loaded",
            "fasttext": "loaded"
        },
        "knowledge_base": {
            "status": "loaded",
            "documents": len(knowledge_base)
        }
    }

# --------------------------------------------------
# Search Endpoint
# --------------------------------------------------

@app.get("/search")
def search(
    query: str = Query(
        ...,
        min_length=1,
        description="Natural language search query"
    ),
    top_k: int = Query(
        5,
        ge=1,
        le=50,
        description="Number of results to return"
    ),
    category: str | None = Query(
        None,
        description="Optional document category"
    )
):
    """
    Perform semantic search using Word2Vec Skip-Gram.
    """

    results = semantic_search(
        query=query,
        model=word2vec_skipgram,
        document_vectors=document_vectors_skipgram,
        knowledge_base=knowledge_base,
        top_k=top_k,
        category=category
    )

    return {
        "query": query,
        "model": "word2vec_skipgram",
        "category": category or "All",
        "results": [
            {
                "document_id": row["document_id"],
                "category": row["category"],
                "title": row["title"],
                "content": row["content"],
                "keywords": row["keywords"],
                "similarity": float(row["similarity"])
            }
            for _, row in results.iterrows()
        ]
    }

# --------------------------------------------------
# Categories Endpoint
# --------------------------------------------------

@app.get("/categories")
def get_categories():

    categories = (
        knowledge_base["category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    categories = sorted(categories)

    return {
        "categories": [
            "All",
            *categories
        ]
    }