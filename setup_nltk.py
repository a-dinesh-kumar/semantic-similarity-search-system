import os
import nltk

# Store NLTK data inside the project
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")

os.makedirs(NLTK_DATA_DIR, exist_ok=True)

packages = [
    "stopwords",
    "punkt",
    "punkt_tab",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]

for package in packages:
    print(f"Downloading/checking NLTK package: {package}")
    nltk.download(package, download_dir=NLTK_DATA_DIR)

print(f"NLTK data stored at: {NLTK_DATA_DIR}")
print("NLTK data setup completed successfully.")