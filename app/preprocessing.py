import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------
# NLP resources
# --------------------------------------------------

NLTK_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "nltk_data"
)

nltk.data.path.insert(0, NLTK_DATA_DIR)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def preprocess_text(text):
    """
    Clean and normalize text for semantic search.
    """

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove special characters and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal
    tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]

    return " ".join(tokens)