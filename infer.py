import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


embedding_model_name = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name)


index_path = "drug_corpus.index"
corpus_path = "drug_corpus.json"

print("Loading FAISS index...")
index = faiss.read_index(index_path)

print("Loading corpus...")
with open(corpus_path, 'r') as f:
    corpus = json.load(f)


llm_model_name = "meta-llama/Llama-3.2-1B"  
print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForCausalLM.from_pretrained(llm_model_name)


def search(query, k=5):
    print("Generating query embedding...")
    query_embedding = embedding_model.encode([query])
    print("Searching FAISS index...")
    distances, indices = index.search(np.array(query_embedding), k)
    retrieved_docs = [corpus[i] for i in indices[0]]
    return retrieved_docs


# Function to generate a response from the LLM
def generate_response(context, question, max_tokens=512):
    print("Generating response from LLM...")
    tokenizer.model_max_length = max_tokens  # Adjust the max token length if necessary
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    # Truncate the context to fit within the model's maximum input size
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_tokens)
    
    outputs = model.generate(**inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def rag_pipeline(query, top_k=5):
    
    retrieved_docs = search(query, k=top_k)
    combined_context = "\n".join(retrieved_docs)
    
    answer = generate_response(combined_context, query)
    return answer


if __name__ == "__main__":
    print("Welcome to the Drug Discovery RAG System!")
    print("Type your query below (or 'exit' to quit).")

    while True:
        user_query = input("\nEnter your query: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = rag_pipeline(user_query)
            print("\nResponse:\n", response)
        except Exception as e:
            print("\nAn error occurred:", e)

