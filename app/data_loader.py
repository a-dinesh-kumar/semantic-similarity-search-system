from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "knowledge_base_processed.csv"
)


# --------------------------------------------------
# Load processed knowledge base
# --------------------------------------------------

def load_knowledge_base():

    print("Loading processed knowledge base...")

    knowledge_base = pd.read_csv(DATA_PATH)

    print(
        f"✓ Knowledge base loaded: "
        f"{len(knowledge_base):,} documents"
    )

    # --------------------------------------------------
    # Validate required columns
    # --------------------------------------------------

    required_columns = [
        "document_id",
        "category",
        "title",
        "content",
        "keywords",
        "cleaned_text"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in knowledge_base.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("✓ Required columns validated")

    # --------------------------------------------------
    # Remove empty documents
    # --------------------------------------------------

    knowledge_base = knowledge_base[
        knowledge_base["cleaned_text"].notna()
        & (
            knowledge_base["cleaned_text"]
            .str.strip()
            != ""
        )
    ].copy()

    knowledge_base = knowledge_base.reset_index(
        drop=True
    )

    print(
        f"✓ Usable documents: "
        f"{len(knowledge_base):,}"
    )

    return knowledge_base