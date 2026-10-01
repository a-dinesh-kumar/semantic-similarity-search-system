import nltk


resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng"
]


for resource in resources:
    print(f"Downloading: {resource}")
    nltk.download(resource)


print("\nAll NLTK resources downloaded successfully.")