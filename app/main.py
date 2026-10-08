from pathlib import Path
from fastapi import FastAPI, Query
from gensim.models import Word2Vec, FastText
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.data_loader import load_knowledge_base
from app.search import (
    get_document_vectors,
    get_query_token_status,
    semantic_search
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

CBOW_MODEL_PATH = MODEL_DIR / "word2vec_cbow.model"
SKIPGRAM_MODEL_PATH = MODEL_DIR / "word2vec_skipgram.model"
FASTTEXT_MODEL_PATH = MODEL_DIR / "fasttext.model"


# ============================================================
# LOAD EMBEDDING MODELS
# ============================================================

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


# ------------------------------------------------------------
# FastText is optional for deployment
# ------------------------------------------------------------

fasttext_model = None

try:
    fasttext_model = FastText.load(
        str(FASTTEXT_MODEL_PATH),
        mmap="r"
    )

    print("✓ FastText loaded")

except (FileNotFoundError, ValueError, EOFError) as e:

    print(
        "⚠ FastText model not available. "
        "Continuing with Word2Vec models only."
    )

    print(f"FastText loading error: {e}")


print("Embedding model loading completed.")


# ============================================================
# MODEL REGISTRY
# ============================================================

MODELS = {
    "word2vec_cbow": word2vec_cbow,
    "word2vec_skipgram": word2vec_skipgram
}

MODEL_LABELS = {
    "word2vec_cbow": "Word2Vec CBOW",
    "word2vec_skipgram": "Word2Vec Skip-Gram"
}


# Add FastText only when it is successfully available
if fasttext_model is not None:

    MODELS["fasttext"] = fasttext_model

    MODEL_LABELS["fasttext"] = "FastText"


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

knowledge_base = load_knowledge_base()


# ============================================================
# PRE-BUILD SELECTED MODEL INDEX
# ============================================================

print("Building Skip-Gram document vectors...")

document_vectors_skipgram = get_document_vectors(
    model_name="word2vec_skipgram",
    model=word2vec_skipgram,
    documents=knowledge_base["cleaned_text"]
)

print(
    f"✓ Skip-Gram document vectors ready: "
    f"{document_vectors_skipgram.shape}"
)

print("Document vectorization completed.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Semantic Similarity Search API",
    description="Semantic search using Word2Vec and FastText embeddings.",
    version="1.0.0"
)


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",

        "models": {
            model_name: "loaded"
            for model_name in MODELS
        },

        "knowledge_base": {
            "status": "loaded",
            "documents": len(knowledge_base)
        }
    }


# ============================================================
# STATS
# ============================================================

@app.get("/stats")
def stats():

    categories = (
        knowledge_base["category"]
        .dropna()
        .astype(str)
    )

    category_counts = (
        categories
        .value_counts()
        .sort_index()
    )

    category_data = [
        {
            "name": category,
            "count": int(count)
        }
        for category, count in category_counts.items()
    ]

    model_data = [
        {
            "id": "word2vec_skipgram",
            "label": "Skip-Gram",
            "vocab": len(word2vec_skipgram.wv)
        },
        {
            "id": "word2vec_cbow",
            "label": "CBOW",
            "vocab": len(word2vec_cbow.wv)
        }
    ]

    if fasttext_model is not None:

        model_data.append(
            {
                "id": "fasttext",
                "label": "FastText",
                "vocab": len(fasttext_model.wv)
            }
        )

    return {
        "ready": True,
        "message": "Semantic search system is ready.",
        "documents": len(knowledge_base),
        "categories": category_data,
        "vector_size": word2vec_skipgram.vector_size,
        "models": model_data
    }


# ============================================================
# MODELS
# ============================================================

@app.get("/models")
def get_models():

    model_data = []

    for model_name, model in MODELS.items():

        label = MODEL_LABELS[model_name]

        model_data.append(
            {
                "id": model_name,
                "label": label,
                "vocab": len(model.wv)
            }
        )

    return {
        "models": model_data
    }


# ============================================================
# CATEGORIES
# ============================================================

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
        "categories": ["All", *categories]
    }


# ============================================================
# SEARCH
# ============================================================

@app.get("/search")
def search(
    query: str = Query(
        ...,
        min_length=1,
        description="Natural language search query"
    ),

    model: str = Query(
        "word2vec_skipgram",
        description="Embedding model"
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

    if model not in MODELS:

        return {
            "error": f"Unsupported model: {model}"
        }

    selected_model = MODELS[model]

    document_vectors = get_document_vectors(
        model_name=model,
        model=selected_model,
        documents=knowledge_base["cleaned_text"]
    )

    query_info = get_query_token_status(
        query=query,
        model=selected_model
    )

    if not query_info["processed_query"]:

        return {
            "query": query,
            "model": model,
            "model_label": MODEL_LABELS[model],
            "processed_query": "",
            "known_tokens": [],
            "unknown_tokens": [],
            "results": []
        }

    results = semantic_search(
        query=query,
        model=selected_model,
        document_vectors=document_vectors,
        knowledge_base=knowledge_base,
        top_k=top_k,
        category=category
    )

    formatted_results = []

    for rank, (_, row) in enumerate(
        results.iterrows(),
        start=1
    ):

        formatted_results.append(
            {
                "rank": rank,
                "document_id": row["document_id"],
                "category": row["category"],
                "title": row["title"],
                "content": row["content"],
                "keywords": row["keywords"],
                "similarity_score": float(row["similarity"])
            }
        )

    return {
        "query": query,
        "model": model,
        "model_label": MODEL_LABELS[model],
        "processed_query": query_info["processed_query"],
        "known_tokens": query_info["known_tokens"],
        "unknown_tokens": query_info["unknown_tokens"],
        "results": formatted_results
    }


# ============================================================
# COMPARE ALL MODELS
# ============================================================

@app.get("/compare")
def compare_models(
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

    comparison_results = {}

    for model_name, model in MODELS.items():

        document_vectors = get_document_vectors(
            model_name=model_name,
            model=model,
            documents=knowledge_base["cleaned_text"]
        )

        query_info = get_query_token_status(
            query=query,
            model=model
        )

        if not query_info["processed_query"]:

            comparison_results[model_name] = {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "processed_query": "",
                "known_tokens": [],
                "unknown_tokens": [],
                "results": []
            }

            continue

        results = semantic_search(
            query=query,
            model=model,
            document_vectors=document_vectors,
            knowledge_base=knowledge_base,
            top_k=top_k,
            category=category
        )

        formatted_results = []

        for rank, (_, row) in enumerate(
            results.iterrows(),
            start=1
        ):

            formatted_results.append(
                {
                    "rank": rank,
                    "document_id": row["document_id"],
                    "category": row["category"],
                    "title": row["title"],
                    "content": row["content"],
                    "keywords": row["keywords"],
                    "similarity_score": float(row["similarity"])
                }
            )

        comparison_results[model_name] = {
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "processed_query": query_info["processed_query"],
            "known_tokens": query_info["known_tokens"],
            "unknown_tokens": query_info["unknown_tokens"],
            "results": formatted_results
        }

    return comparison_results