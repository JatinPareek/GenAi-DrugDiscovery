from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import torch
from transformers import pipeline
import re
import json
import requests
from bs4 import BeautifulSoup
from Bio import Entrez
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


Entrez.email = "your_email@example.com"


def fetch_pubmed(query, max_results=400):
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


def scrape_kegg_pathways():
    url = "https://www.genome.jp/kegg/pathway.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    pathways = []
    for link in soup.select("a[href*='/pathway/map']"):
        title = link.text
        pathways.append(title)

    return pathways


def create_corpus():
    print("Fetching PubMed abstracts...")
    pubmed_articles = fetch_pubmed("drug discovery", max_results=400)
    print(f"Fetched {len(pubmed_articles)} abstracts.")

    print("Scraping KEGG pathways...")
    kegg_pathways = scrape_kegg_pathways()
    print(f"Scraped {len(kegg_pathways)} pathways.")

    return pubmed_articles + kegg_pathways


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def preprocess_corpus(corpus):
    return [clean_text(doc) for doc in corpus]


def save_corpus(corpus, file_path="drug_corpus.json"):
    print(f"Saving corpus to {file_path}...")
    with open(file_path, 'w') as f:
        json.dump(corpus, f)


summarizer = pipeline("summarization", model="google/pegasus-xsum")


def build_search_engine(corpus, embedding_model_name="all-MiniLM-L6-v2"):
    print("Generating embeddings...")
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode(corpus, show_progress_bar=True)
    print("Building FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))
    faiss.write_index(index, 'drug_corpus.index')
    return index, embedding_model


def search(query, index, corpus, embedding_model, k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [corpus[i] for i in indices[0]]


def chunk_text(text, max_length=500):
    """Split text into chunks that fit within model's maximum length."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def trim_context(context, max_tokens=500):
    """Trim context to a manageable size while preserving meaning."""
    sentences = context.split(".")
    trimmed_sentences = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence.split()) > max_tokens:
            break
        trimmed_sentences.append(sentence)
        current_length += len(sentence.split())
    
    return ". ".join(trimmed_sentences) + "."


def summarize_context(retrieved_docs, max_chunk_length=500):
    """Summarize context with chunking for long texts."""
    # Combine and clean the retrieved documents
    combined_context = " ".join(retrieved_docs)
    
    # Split into chunks if too long
    chunks = chunk_text(combined_context, max_chunk_length)
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=50, min_length=30, do_sample=False)
            summaries.append(summary[0]["summary_text"])
        except Exception as e:
            print(f"Warning: Summarization failed for chunk: {e}")
            # Fall back to extractive summarization for this chunk
            sentences = chunk.split('.')[:5]  # Take first 3 sentences as fallback
            summaries.append('. '.join(sentences))
    
    # Combine summaries
    return " ".join(summaries)


def rag_system(query, index, corpus, embedding_model, model_name="meta-llama/Llama-3.2-1B"):
    """Enhanced RAG system with better error handling and length management."""
    try:
        # Get relevant documents
        retrieved_docs = search(query, index, corpus, embedding_model)
        
        # Trim context to manageable size
        trimmed = trim_context("\n".join(retrieved_docs), max_tokens=400)
        
        # Generate summary with chunking
        summarized = summarize_context([trimmed], max_chunk_length=500)
        
        # Ensure final context isn't too long for the LLM
        if len(summarized.split()) > 450:  # Leave room for query and prompt
            summarized = " ".join(summarized.split()[:450])
        
        # Generate response
        response = generate_response(summarized, query, model_name)
        return response
        
    except Exception as e:
        print(f"Error in RAG system: {e}")
        # Fallback to simpler response using just the first retrieved document
        try:
            first_doc = search(query, index, corpus, embedding_model, k=1)[0]
            trimmed_doc = trim_context(first_doc, max_tokens=200)
            return generate_response(trimmed_doc, query, model_name)
        except Exception as e:
            return f"Unable to generate response: {str(e)}"


def generate_response(context, question, model_name="meta-llama/Llama-3.2-1B"):
    """Enhanced response generation with length checks."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Construct input with length checking
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    try:
        outputs = model.generate(
            **inputs,
            max_length=1500,
            min_length=50,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error in response generation: {e}")
        return "Unable to generate response due to an error."
class MoleculeGenerator:
    def __init__(self, model_name="ncfrey/ChemGPT-19M"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def generate_smiles(self, prompt, num_sequences=5):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=num_sequences,
            num_beams=num_sequences,
            temperature=0.7,
            do_sample=True
        )
        
        generated_smiles = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        return self._filter_valid_smiles(generated_smiles)
    
    def _filter_valid_smiles(self, smiles_list):
        valid_smiles = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_smiles.append(smiles)
            except:
                continue
        return valid_smiles

def analyze_molecule(smiles):
    """Analyze molecular properties of generated SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return {
        'molecular_weight': Descriptors.ExactMolWt(mol),
        'logP': Descriptors.MolLogP(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol)
    }

if __name__ == "__main__":
    print("Creating corpus...")
    raw_corpus = create_corpus()

    print("Cleaning and preprocessing corpus...")
    cleaned_corpus = preprocess_corpus(raw_corpus)
    save_corpus(cleaned_corpus)

    print("Building dense search engine...")
    faiss_index, embed_model = build_search_engine(cleaned_corpus)

    print("Starting enhanced drug discovery pipeline...")
    
    # 1. Get symptoms and generate initial analysis
    symptoms = input("What are the symptoms? ")
    initial_query = f"""Give medically valid, concise answers. 
    What are the key causes for the symptoms: {symptoms}?"""

    causes_answer = rag_system(initial_query, faiss_index, cleaned_corpus, embed_model)
    print(f"\nCauses Analysis: {causes_answer}")
    
    # 2. Generate treatment suggestions
    causes_summary = summarize_context([causes_answer])
    treatment_query = f"What biological targets and treatment approaches would be most effective for: {causes_summary}"
    treatment_answer = rag_system(treatment_query, faiss_index, cleaned_corpus, embed_model)
    print(f"\nTreatment Analysis: {treatment_answer}")
    
    # 3. Generate drug candidates using SMILES
    print("\nGenerating potential drug candidates...")
    mol_gen = MoleculeGenerator()
    
    # Create a prompt based on the treatment analysis
    drug_prompt = f"""
    Based on the following treatment approach:
    {summarize_context([treatment_answer])}
    
    Generate SMILES strings for drug-like molecules that:
    - Target the identified biological pathways
    - Have good drug-likeness properties
    - Are synthetically accessible
    """
    
    # Generate and analyze drug candidates
    generated_smiles = mol_gen.generate_smiles(drug_prompt)
    print("\nGenerated Drug Candidates:")
    
    for i, smiles in enumerate(generated_smiles, 1):
        properties = analyze_molecule(smiles)
        if properties:
            print(f"\nCandidate {i}:")
            print(f"SMILES: {smiles}")
            print("Properties:")
            for prop, value in properties.items():
                print(f"- {prop}: {value:.2f}")
            
            # Check Lipinski's Rule of Five
            lipinski_violations = 0
            if properties['molecular_weight'] > 500: lipinski_violations += 1
            if properties['logP'] > 5: lipinski_violations += 1
            if properties['HBA'] > 10: lipinski_violations += 1
            if properties['HBD'] > 5: lipinski_violations += 1
            
            print(f"Lipinski violations: {lipinski_violations}")
    
    print("\nNext steps:")
    print("1. Validate generated molecules through docking simulations")
    print("2. Analyze protein-ligand interactions")
    print("3. Assess synthetic accessibility")
    print("4. Consider ADMET properties")
    print("5. Plan initial synthesis routes")
