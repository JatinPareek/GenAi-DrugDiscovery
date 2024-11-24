Here’s a comprehensive `README.md` for your project:

---

# **Drug Discovery RAG System**

This project implements a **Retrieval-Augmented Generation (RAG)** system for drug discovery, drug pathways, and pharmaceutical knowledge. It combines:
- **Corpus creation**: Collects data from PubMed and KEGG Pathways.
- **Dense Search Engine**: Uses SentenceTransformers and FAISS for efficient similarity-based search.
- **RAG Integration**: Integrates the Llama-3.2-1B model for answering complex queries by retrieving and processing domain-specific information.

---

## **Features**
1. **Automated Corpus Creation**:
   - Fetches abstracts from PubMed.
   - Scrapes pathway information from KEGG.
2. **Text Preprocessing**:
   - Cleans and structures data for use in machine learning pipelines.
3. **Dense Retrieval**:
   - Embeds the corpus using SentenceTransformers and builds a FAISS-based search engine.
4. **LLM Integration**:
   - Uses the Llama-3.2-1B model (via Hugging Face) for generating answers to user queries.

---

## **Requirements**
- Python 3.8+
- GPU (optional but recommended for faster embeddings and LLM inference)
- Dependencies listed in `requirements.txt`

---

## **Setup Instructions**

### 1. **Clone the Repository**
```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. **Install Dependencies**
Ensure you have Python 3.8+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. **Prepare the Corpus**
Run the script to create, preprocess, and build the dense search engine:
```bash
python rag_system.py
```

This will:
- Fetch PubMed abstracts and KEGG pathways.
- Preprocess and clean the raw data.
- Generate embeddings and save the FAISS index (`drug_corpus.index`).

### 4. **Test the System**
After the setup, you can test the RAG system with a sample query:
```bash
python rag_system.py
```

The script includes a sample query (`What are the key pathways involved in drug metabolism?`) to demonstrate the system's functionality.

---

## **Project Structure**
```
.
├── rag_system.py          # Main script to create corpus, build search engine, and test RAG
├── requirements.txt       # Dependencies for the project
├── README.md              # Project documentation
├── drug_corpus.json       # (Auto-generated) JSON file containing the preprocessed corpus
├── drug_corpus.index      # (Auto-generated) FAISS index for dense search
```

---

## **Usage**

### Run Custom Queries
Modify the `test_query` variable in `rag_system.py` to input your own query:
```python
test_query = "Your custom question here"
```

### Add Additional Sources
- Extend the `create_corpus()` function in `rag_system.py` to fetch or scrape additional sources of knowledge.

---

## **Technical Details**

### Dense Search
- **Model**: `all-MiniLM-L6-v2` (from SentenceTransformers).
- **Storage**: FAISS index stored as `drug_corpus.index`.

### Language Model
- **Model**: Llama-3.2-1B (via Hugging Face Transformers).
- **Inference**: Retrieves context using the dense search engine, feeds it as input to the LLM for contextual answering.

---

## **System Requirements**
### Minimal
- **RAM**: 8–16 GB
- **CPU**: Multi-core processor
- **Storage**: ~10 GB (corpus, index, and model files)

### Recommended
- **RAM**: 16–32 GB
- **GPU**: NVIDIA with 16 GB VRAM (for faster embeddings and LLM inference)
- **Storage**: ~10–20 GB

