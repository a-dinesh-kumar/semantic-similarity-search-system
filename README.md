# Semantic Similarity Search System

A semantic search application that finds relevant documents based on **meaning and context**, rather than relying only on exact keyword matching.

The system preprocesses natural-language queries, converts documents and queries into dense vector representations using **Word2Vec embeddings**, and ranks documents using **cosine similarity**.

---
## 🚀 Live application

https://semantic-similarity-search-system-5u4u.onrender.com/

---

## 🚀 Project Overview

Traditional keyword-based search mainly looks for matching words. 

For example, a user may search:
> "I forgot my password and cannot access my account"

A keyword-based system may depend heavily on words such as `password` or `account`. A semantic search system can understand the meaning of the query and retrieve a document such as:
> "Regain Access to Your Password"

even when the wording is different.

### Search Pipeline

```text
User Query
     ↓
Text Preprocessing
     ↓
Query Vector
     ↓
Cosine Similarity
     ↓
Document Ranking
     ↓
Top-K Results
```

---

## 🎯 Project Objectives

The main objectives of this project are:
* Build a semantic search system using word embeddings
* Understand and implement text preprocessing
* Train and compare multiple embedding models
* Convert documents into dense vector representations
* Convert user queries into vector representations
* Calculate semantic similarity using cosine similarity
* Retrieve the most relevant Top-K documents
* Provide category-based filtering
* Handle unknown words in queries
* Expose the ML system through a FastAPI backend
* Build a browser-based frontend for interacting with the search system
* Deploy the trained NLP models into a usable application architecture

---

## ✨ Key Features

* **Core Search Engine**
  * Semantic search based on document meaning
  * Cosine similarity based ranking
  * Configurable Top-K results
  * Category filtering
* **Embedding Model Support**
  * Word2Vec CBOW support
  * Word2Vec Skip-Gram support (Selected as the current primary model)
  * FastText support
  * Mean pooling for document-level embeddings
  * Known-token and unknown-token visibility
  * Single-model search vs. Compare-all-models search
* **Text Preprocessing**
  * Tokenization
  * Stop-word removal
  * Lemmatization
* **API Backend (FastAPI)**
  * Swagger/OpenAPI documentation
  * Cached document vectors
  * Health-check and statistics endpoints
  * Model information and category endpoints
* **Frontend Design**
  * Interactive UI with a light theme by default (Dark theme support included)
  * Responsive frontend design
  * Quick links to GitHub and LinkedIn

---

## 🧠 Machine Learning Approach

The project uses Word2Vec embeddings to convert words into dense numerical vectors. The trained models understand relationships between words based on their surrounding context.

```text
password  →  authentication  →  login  →  account
```

Words that occur in similar contexts tend to have similar vector representations. The project trains and supports three embedding approaches:
1. Word2Vec CBOW
2. Word2Vec Skip-Gram
3. FastText

After experimentation, **Word2Vec Skip-Gram** was selected as the current primary model.

### 🔬 Model Comparison

| Model | Description | Status |
| :--- | :--- | :--- |
| **Word2Vec CBOW** | Predicts a target word from surrounding context | Available |
| **Word2Vec Skip-Gram** | Predicts surrounding words from a target word | **Selected** |
| **FastText** | Uses word and subword information | Available |

The application allows the user to switch between the models and compare their search results.

### ⭐ Why Word2Vec Skip-Gram?

The project currently uses **Word2Vec Skip-Gram** as the primary embedding model. Skip-Gram learns word representations by using a target word to predict surrounding context words.

```text
        context word
              ↑
              |
context ← target word → context
              |
              ↓
        context word
```

This makes it useful for learning relationships between words from their context. The other trained models remain available in the application for comparison and experimentation.

---

## 📊 Dataset

The system uses a knowledge base containing approximately **50,000 documents**. The dataset contains the following fields:

| Column | Description |
| :--- | :--- |
| `document_id` | Unique document identifier |
| `category` | Document category |
| `title` | Document title |
| `content` | Main document content |
| `keywords` | Related keywords |

---

## 🧹 Text Preprocessing

Before training the embedding models, the text goes through a preprocessing pipeline:
1. Convert text to lowercase
2. Remove URLs
3. Remove non-alphabetic characters
4. Normalize whitespace
5. Tokenize the text
6. Remove English stop words
7. Apply WordNet lemmatization

### Example
* **Input:** *I forgot my password and could not access my account.*
* **After preprocessing:** *forgot password could access account*

The exact same preprocessing function is used for both **knowledge-base documents** and **user search queries** to keep representations structurally consistent.

### 🗂️ Combined Document Text

The fields `title`, `content`, and `keywords` are combined before running the text pipeline:

```text
Title + Content + Keywords
           ↓
     Combined Text
           ↓
     Preprocessing
           ↓
     Cleaned Text
```

The cleaned text is then used directly for embedding training and document vectorization.

---

## 📚 Train/Test Split

The knowledge base was divided into an **80% Training / 20% Testing** split at the document level (not at the individual word level).

```text
               50,000 Documents
                       │
        ┌──────────────┴──────────────┐
        ↓                             ↓
   80% Training                  20% Testing
        │                             │
        ↓                             ↓
Train Embeddings                Generalisation
```

The training documents are used to train the embedding models, while the held-out test documents remain available for evaluation and experimentation.

---

## 🔢 Word2Vec Configuration

The embedding models were trained with the following common hyperparameter configuration:
* **Vector Size** : 100
* **Window Size** : 5
* **Min Count**   : 2
* **Epochs**      : 10
* **Workers**     : 4
* **Seed**        : 42

*For architecture assignment in Gensim: CBOW is defined with `sg=0`, and Skip-Gram with `sg=1`.*

---

## 📐 Document Vectorization

Word2Vec creates a vector for each individual word, but a document contains multiple words. This project uses **mean pooling** to convert an entire document into a single vector space.

### Example
For a document containing: `"forgot password account access"`

```text
forgot   → [ ... 100 values ... ]
password → [ ... 100 values ... ]
account  → [ ... 100 values ... ]
access   → [ ... 100 values ... ]
             ↓
        Mean Pooling
             ↓
      Document Vector (100 dimensions)
```

The final document index representation matrix holds a shape of **50,000 × 100** for the complete knowledge base.

---

## 🔍 Query Vectorization

When a user submits a search query, it passes through the same vectorization pipeline to maintain matrix harmony:

```text
Raw Query → Clean & Tokenize → Remove Stop Words & Lemmatize → Word Embeddings → Mean Pooling → Query Vector
```

---

## 📏 Cosine Similarity & Retrieval

The system uses cosine similarity to determine how closely aligned the query vector is to each indexed document vector.

```text
                Query Vector
                     ↓
              ┌──────┴──────┐
              ↓             ↓
        Document Vector  Document Vector
              ↓             ↓
              └──────┬──────┘
                     ↓
             Cosine Similarity
                     ↓
                  Ranking
```

A higher cosine similarity score indicates that the vectors are more closely aligned. The documents are sorted based on their similarity score.

### 🏆 Top-K Retrieval & Category Filtering
* **Top-K Retrieval:** Supports configurable Top-K retrieval (e.g., returning the 5 most similar documents). The frontend allows adjustments, while the backend caps the query limit at **50 results**.
* **Category Filtering:** Users can choose to filter search results by a specific bucket (e.g., `Authentication`). The system will then exclusively run semantic evaluations against documents belonging to that categorical partition.
---

## 🔤 Unknown Token Handling

The application provides real-time visibility into how search queries are parsed by the models. After preprocessing, tokens are classified into two groups:
* **Known Tokens:** Tokens explicitly present in the selected model's vocabulary matrix.
* **Unknown Tokens:** Out-of-vocabulary (OOV) tokens that are missing from the training vocabulary.

### Example
For a processed query string like `"forgot password reset account"`, the user interface displays the token states separately:
```text
Known Tokens:
  - forgot
  - password
  - reset
  - account

Unknown Tokens:
  - [None] (or individual OOV terms listed when applicable)
```
This validation panel helps developers and users understand how the query is being interpreted by the mathematical embedding layer before retrieval occurs.

---

## 🤖 Model Evaluation Modes

The application supports two primary search search operations:
1. **Single Model Mode:** The user selects one specific embedding archetype (e.g., `Word2Vec Skip-Gram`) and retrieves its respective Top-K semantic document matches.
2. **Compare All 3 Mode:** The system runs the input query simultaneously against all three architectures (`Word2Vec CBOW`, `Word2Vec Skip-Gram`, and `FastText`) and renders the results side-by-side for comparative testing.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │      Web Browser     │
                    │                      │
                    │   HTML / CSS / JS    │
                    └──────────┬───────────┘
                               │
                               │ HTTP
                               ↓
                    ┌──────────────────────┐
                    │       FastAPI        │
                    │       Backend        │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼───────────────┐
                ↓              ↓               ↓
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │ Preprocess  │ │   Search    │ │ Data Loader │
        │             │ │             │ │             │
        └─────────────┘ └──────┬──────┘ └─────────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │   Embedding Models   │
                    │                      │
                    │ Word2Vec CBOW        │
                    │ Word2Vec Skip-Gram   │
                    │ FastText             │
                    └──────────┬───────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │   Document Vectors   │
                    │                      │
                    │    50,000 × 100      │
                    └──────────┬───────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │ Cosine Similarity    │
                    │        +             │
                    │   Top-K Ranking      │
                    └──────────────────────┘
```

---

## 📁 Project Structure

```text
semantic-similarity-search-system/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── search.py
│
├── models/
│   ├── word2vec_cbow.model
│   ├── word2vec_skipgram.model
│   └── fasttext.model
│
├── static/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── requirements.txt
└── README.md
```

---

## 🧩 Backend Components

* **`main.py`**
  Responsible for core application orchestration. It handles FastAPI framework initialization, loads the pre-trained embedding models, mounts the static assets/frontend pages, configures routes, and hooks incoming HTTP requests to the underlying search logic.
* **`preprocessing.py`**
  Contains the centralized, reusable text preprocessing function. Applying the exact same preprocessing pipeline to both the baseline knowledge-base corpus and dynamic user queries prevents data skew.
* **`search.py`**
  Houses the vector arithmetic and semantic calculation engines. Key functions include:
  * `get_document_vector()`: Converts a single document into an averaged vector.
  * `build_document_vectors()`: Compiles arrays for multiple documents.
  * `get_document_vectors()`: Manages the vector memory cache to bypass rebuild steps.
  * `get_query_vector()`: Preprocesses a live text query and translates it to a vector space.
  * `get_query_token_status()`: Interrogates vocabulary matrices to find known/unknown terms.
  * `semantic_search()`: Executes the full calculation pipeline: `Query Vector → Cosine Similarity Calculation → Category Filtering → Sorting → Top-K Extraction`.
* **`data_loader.py`**
  Handles localized filesystem access to properly unpack and structure the document database.

---

## ⚡ Document Vector Caching

Generating dense vector arrays for **50,000 documents** dynamically on every single search call would devastate API performance. To optimize latency, the application implements an in-memory vector cache matrix:

```text
50,000 Documents  →  Vector Generation Engine  →  50,000 × 100 Matrix  →  Stored in Server RAM
```

The very first query triggered against a model compiles the underlying matrix. Subsequent searches reuse this in-memory matrix, dropping search latency down to milliseconds.

---

## 🌐 API Endpoints

### 1. Health Check
* **Protocol / Path:** `GET /health`
* **Purpose:** Validates engine up-time, model statuses, and database connections.
* **Response Example:**
  ```json
  { "status": "healthy" }
  ```

### 2. System Statistics
* **Protocol / Path:** `GET /stats`
* **Purpose:** Returns comprehensive operational counters (e.g., number of indexed documents, total unique categories, vector dimensions, active architectures).

### 3. Model Information
* **Protocol / Path:** `GET /models`
* **Purpose:** Lists all vector models currently registered in the system.

### 4. Categorical Metadata
* **Protocol / Path:** `GET /categories`
* **Purpose:** Returns available structural labels for search space filtering.

### 5. Semantic Search Lookup
* **Protocol / Path:** `GET /search`
* **Sample Request Route:** `/search?query=forgot%20my%20password&model=word2vec_skipgram&top_k=5`
* **Query Parameters:**

  | Parameter | Type | Description |
  | :--- | :--- | :--- |
  | `query` | String | Raw textual search expression submitted by the user. |
  | `model` | String | Identifier of target model (`word2vec_cbow`, `word2vec_skipgram`, `fasttext`). |
  | `top_k` | Integer | Limits retrieved document match boundaries (Max limit: 50). |
  | `category` | String | *Optional.* Restricts cosine comparison to this exact partition. |

### 6. Parallel Model Comparison
* **Protocol / Path:** `GET /compare`
* **Sample Request Route:** `/compare?query=track%20my%20order&top_k=5`
* **Purpose:** Runs a single query concurrently against all three architectures and groups results cleanly into separate JSON nodes.

---

## 🖥️ Frontend Operations & UI

The browser client is a decoupled, modern interface written using semantic HTML5, clean CSS3 properties, and Vanilla JavaScript. It interfaces with the FastAPI backend using standard Web API Fetch workflows.

* **Main Features:** Parameter control panels (model switches, category pickers, Top-K sliders), statistical data cards, dark theme state persistence (via browser `localStorage`), responsive UI layouts, and live vocabulary classification views.
* **📤 CSV Upload Management:** The original UI wireframe references an interactive CSV upload widget. However, production model retraining operations are intentionally deactivated here to focus server resources entirely on fast retrieval execution rather than expensive pipeline updates.

---

## 🛠️ Technology Stack

* **Core Machine Learning:** Python, Gensim, Scikit-learn, Pandas, NumPy, Joblib
* **Natural Language Processing:** NLTK (Tokenization, Stop-words filtering, WordNet Lemmatizer)
* **Backend Framework:** FastAPI, Uvicorn ASGI Server
* **Frontend Components:** Native JavaScript, CSS variables, Font Awesome UI components

---

## 📦 Environment & Dependency Specifications

This pipeline was built and frozen under the following system version specifications. Ensure matching runtimes locally to prevent computation mismatch anomalies:

* **Runtime:** Python `3.13.15`
* **Package Specifications (`requirements.txt`):**
  ```text
  numpy==2.1.3
  scipy==1.16.3
  pandas==2.2.3
  scikit-learn==1.6.1
  gensim==4.4.0
  nltk==3.9.1
  joblib==1.6.0
  fastapi==0.141.1
  uvicorn==0.52.4
  ```
---

## 📚 NLTK Resources

The following NLTK resources are required:
* `punkt`
* `punkt_tab`
* `stopwords`
* `wordnet`
* `omw-1.4`
* `averaged_perceptron_tagger`
* `averaged_perceptron_tagger_eng`

They can be downloaded using:

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com
```

Move into the project directory:
```bash
cd semantic-similarity-search-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

**Windows Activation:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the FastAPI server from the project root:
```bash
uvicorn app.main:app --reload
```

The application will be available at:
* [http://127.0.0.1:8000](http://127.0.0.1:8000)

Open the browser:
* [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📖 Swagger API Documentation

FastAPI automatically provides interactive API documentation. Open:
* [http://127.0.0](http://127.0.0)

The Swagger interface can be used to test:
* `/health`
* `/stats`
* `/models`
* `/categories`
* `/search`
* `/compare`

---

## 🔎 Example Searches

Try queries such as:
* *forgot my password*
* *track my order*
* *refund for damaged item*
* *change delivery address*
* *I cannot access my account*

The system attempts to retrieve documents based on semantic similarity rather than requiring exact wording.

---

## 🧪 Error and Empty Query Handling

The application handles empty search input. If the user presses Search without entering a query, the application displays:
> Please enter a search query.

The query field receives focus so the user can immediately enter a search term.

The system also handles cases where preprocessing removes everything from the query. For example, if a query contains only stop words or otherwise leaves no searchable tokens after preprocessing, the application reports that no searchable words remained.

---

## 📈 Current System Statistics

The current knowledge base contains approximately:
* **Documents**          : 50,000
* **Categories**         : 15
* **Vector Dimensions**  : 100
* **Embedding Models**   : 3

Available models:
* Word2Vec Skip-Gram
* Word2Vec CBOW
* FastText

---

## ⚠️ Current Limitations

The current implementation intentionally focuses on the core semantic-search pipeline and has a few architectural constraints:

1. **Mean Pooling:** The document vector is created by averaging word vectors. This is simple and efficient, but it does not capture complex sentence-level relationships.
2. **Word2Vec Context:** Word2Vec provides word-level contextual representations but does not understand complete sentences in the same way as modern transformer-based sentence embeddings.
3. **Evaluation:** The current evaluation/experimentation approach is not a human-annotated semantic relevance benchmark. A production-grade evaluation would require a proper set of: `Query + Relevant Documents + Human Relevance Labels`.
4. **In-Memory Search:** The current implementation compares the query against the document-vector matrix directly. For very large datasets, this approach would eventually become inefficient.
5. **No Vector Database:** The current system does not use a dedicated vector database.
6. **No ANN Index:** The current implementation does not use an approximate nearest-neighbor index such as FAISS.

---

## 🚀 Future Enhancements

1. **Hybrid Word2Vec + FastText:** One future direction is to combine Word2Vec and FastText so that Word2Vec handles the normal vocabulary while FastText handles rare or unseen words.
   ```text
                    Query
                      ↓
                Preprocessing
                      ↓
             ┌────────┴────────┐
             ↓                 ↓
    Word2Vec Skip-Gram      FastText
             │                 │
             └────────┬────────┘
                      ↓
            Hybrid Representation
                      ↓
                Semantic Search
   ```
2. **TF-IDF Baseline:** Introduce TF-IDF as a keyword-based baseline to allow a direct comparison of *Keyword Search VS Semantic Search*. (TF-IDF was omitted here to keep the focus purely on the Word2Vec semantic system).
3. **Transformer-Based Embeddings:** Experiment with Sentence Transformers, transformer-based embeddings, and domain-specific models to transition from `Word2Vec → Sentence Transformers → Modern Semantic Search`.
4. **Vector Database:** Introduce dedicated vector storage technologies for larger knowledge bases, such as FAISS, Qdrant, Milvus, or Elasticsearch/OpenSearch vector search.
5. **Approximate Nearest Neighbor (ANN) Search:** Shift from a brute-force approach (`Query → Compare with 50,000 vectors → Rank`) to an indexing approach (`Query → Vector Index → Nearest Neighbors → Top-K`) to significantly improve system scalability.
6. **Better Evaluation:** Implement a robust metrics framework tracking `Precision@K`, `Recall@K`, `Mean Reciprocal Rank (MRR)`, `NDCG`, and curated benchmark query sets alongside human relevance labels.

---

## 🧠 What I Learned From This Project

This project demonstrates a comprehensive end-to-end NLP and semantic-search workflow, transitioning from raw data processing to user interaction:

```text
Raw Dataset → Data Preparation → Text Preprocessing → Train/Test Split → Embedding Model Training → Model Comparison → Model Selection → Document Vectorization → Query Vectorization → Cosine Similarity → Top-K Retrieval → FastAPI → Frontend
```

It also bridges the gap between isolated machine learning experimentation and production engineering:

```text
Machine Learning Experiment → Reusable Python Functions → Trained Models → FastAPI Backend → Browser UI → Usable AI Application
```

This journey represents a key progression from learning standalone ML/NLP concepts toward engineering complete, user-facing AI applications.

---

### 👨‍💻 Author

**Dinesh Kumar Alagarsamy**  
*Software Quality Engineer / SDET*
-   GitHub: https://github.com/a-dinesh-kumar
-   LinkedIn: https://www.linkedin.com/in/dinesh-kumar-alagarsamy

#### Areas of Interest:
* Quality Engineering
* Test Automation
* API Testing
* Artificial Intelligence
* Machine Learning
* AI-powered Testing
* Software Architecture

------------------------------------------------------------------------

## License

This project is intended for educational, portfolio, and demonstration
purposes.
