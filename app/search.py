import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessing import preprocess_text

DOCUMENT_VECTORS = {}

# --------------------------------------------------
# Document Vectorization
# --------------------------------------------------

def get_document_vector(text, model):
    """
    Convert a document into a single dense vector.

    Each word is represented by its embedding vector.
    The document vector is created by taking the mean
    of all available word vectors.
    """

    tokens = str(text).split()

    vectors = [
        model.wv[token]
        for token in tokens
        if token in model.wv
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def get_document_vectors(
    model_name,
    model,
    documents
):
    """
    Return cached document vectors.

    Build them only when requested for
    the first time.
    """

    if model_name not in DOCUMENT_VECTORS:

        print(
            f"Building document vectors for "
            f"{model_name}..."
        )

        DOCUMENT_VECTORS[model_name] = (
            build_document_vectors(
                documents,
                model
            )
        )

        print(
            f"✓ {model_name} document vectors created: "
            f"{DOCUMENT_VECTORS[model_name].shape}"
        )

    return DOCUMENT_VECTORS[model_name]


# --------------------------------------------------
# Query Vectorization
# --------------------------------------------------

def get_query_vector(query, model):
    """
    Convert a raw user query into a dense vector.
    """

    processed_query  = preprocess_text(query)

    if not processed_query .strip():
        return np.zeros(model.vector_size)

    return get_document_vector(processed_query , model)

def get_query_token_status(query, model):
    """
    Preprocess the query and identify known and unknown tokens.

    Known tokens:
        Tokens available in the model vocabulary.

    Unknown tokens:
        Tokens not present in the model vocabulary.
    """

    processed_query = preprocess_text(query)

    if not processed_query.strip():
        return {
            "processed_query": "",
            "known_tokens": [],
            "unknown_tokens": []
        }

    tokens = processed_query.split()

    known_tokens = [
        token
        for token in tokens
        if token in model.wv.key_to_index
    ]

    unknown_tokens = [
        token
        for token in tokens
        if token not in model.wv.key_to_index
    ]

    return {
        "processed_query": processed_query,
        "known_tokens": known_tokens,
        "unknown_tokens": unknown_tokens
    }


# --------------------------------------------------
# Document Matrix Creation
# --------------------------------------------------

def build_document_vectors(documents, model):
    """
    Convert all documents into dense vectors.
    """

    document_vectors = np.array([
        get_document_vector(text, model)
        for text in documents
    ])

    return document_vectors


# --------------------------------------------------
# Semantic Search
# --------------------------------------------------

def semantic_search(
    query,
    model,
    document_vectors,
    knowledge_base,
    top_k=5,
    category=None
):
    """
    Search the knowledge base using semantic similarity.

    Optional category filtering is applied before
    similarity ranking.
    """

    top_k = max(1, min(top_k, 50))

    # ----------------------------------------------
    # Filter documents
    # ----------------------------------------------

    if category and category.lower() != "all":

        mask = (
            knowledge_base["category"]
            .fillna("")
            .str.lower()
            == category.lower()
        )

        filtered_knowledge_base = (
            knowledge_base[mask]
        )

        filtered_document_vectors = (
            document_vectors[mask.values]
        )

    else:

        filtered_knowledge_base = knowledge_base

        filtered_document_vectors = document_vectors

    # ----------------------------------------------
    # Query vector
    # ----------------------------------------------

    query_vector = get_query_vector(
        query,
        model
    )

    # ----------------------------------------------
    # Cosine similarity
    # ----------------------------------------------

    similarities = cosine_similarity(
        query_vector.reshape(1, -1),
        filtered_document_vectors
    )[0]

    # ----------------------------------------------
    # Ranking
    # ----------------------------------------------

    top_k = min(
        top_k,
        len(filtered_knowledge_base)
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    results = filtered_knowledge_base.iloc[
        top_indices
    ].copy()

    results["similarity"] = similarities[
        top_indices
    ]

    return results

# --------------------------------------------------
# Category Filtering
# --------------------------------------------------

def filter_by_category(
    knowledge_base,
    category=None
):
    """
    Filter the knowledge base by category.

    If category is None or 'All', return the
    complete knowledge base.
    """

    if not category or category.lower() == "all":
        return knowledge_base

    return knowledge_base[
        knowledge_base["category"].str.lower()
        == category.lower()
    ].copy()