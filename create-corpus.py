import re
import json
import requests
from bs4 import BeautifulSoup
from Bio import Entrez
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- STEP 1: FETCH DATA ---- #

# Configure PubMed API
Entrez.email = "your_email@example.com"  # Replace with your email

def fetch_pubmed(query, max_results=100):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]
    articles = []

    for pmid in ids:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        abstract = handle.read()
        articles.append(abstract)
        handle.close()

    return articles

# Scrape KEGG pathways
def scrape_kegg_pathways():
    url = "https://www.genome.jp/kegg/pathway.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    pathways = []
    for link in soup.select("a[href*='/pathway/map']"):
        title = link.text
        pathways.append(title)

    return pathways

# Fetch and combine data
def create_corpus():
    print("Fetching PubMed abstracts...")
    pubmed_articles = fetch_pubmed("drug discovery", max_results=100)
    print(f"Fetched {len(pubmed_articles)} abstracts.")

    print("Scraping KEGG pathways...")
    kegg_pathways = scrape_kegg_pathways()
    print(f"Scraped {len(kegg_pathways)} pathways.")

    return pubmed_articles + kegg_pathways

# ---- STEP 2: CLEAN DATA ---- #

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def preprocess_corpus(corpus):
    return [clean_text(doc) for doc in corpus]

# ---- STEP 3: BUILD DENSE SEARCH ENGINE ---- #

def build_search_engine(corpus, embedding_model_name="all-MiniLM-L6-v2"):
    print("Generating embeddings...")
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode(corpus, show_progress_bar=True)

    print("Building FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))

    # Save the index
    faiss.write_index(index, 'drug_corpus.index')
    return index, embedding_model

# Save the cleaned corpus to a JSON file
def save_corpus(corpus, file_path="drug_corpus.json"):
    print(f"Saving corpus to {file_path}...")
    with open(file_path, 'w') as f:
        json.dump(corpus, f)

# Main script section
if __name__ == "__main__":
    print("Creating corpus...")
    raw_corpus = create_corpus()

    print("Cleaning and preprocessing corpus...")
    cleaned_corpus = preprocess_corpus(raw_corpus)

    # Save the corpus
    save_corpus(cleaned_corpus)

    print("Building dense search engine...")
    faiss_index, embed_model = build_search_engine(cleaned_corpus)

