import json
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

class DrugRAGInference:
    def __init__(
        self,
        corpus_path="drug_corpus.json",
        index_path="drug_corpus.index",
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model_name="meta-llama/Llama-3.2-1B"
    ):
        # Initialize models and load data
        print("Initializing RAG system...")
        self.load_corpus(corpus_path)
        self.load_models(embedding_model_name, llm_model_name)
        self.load_faiss_index(index_path)
        print("System initialized and ready for queries!")

    def load_corpus(self, corpus_path):
        print("Loading corpus...")
        with open(corpus_path, 'r') as f:
            self.corpus = json.load(f)
        print(f"Loaded corpus with {len(self.corpus)} documents")

    def load_models(self, embedding_model_name, llm_model_name):
        print("Loading models...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.summarizer = pipeline("summarization", model="google/pegasus-xsum")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        
        # Move models to GPU if available
        if torch.cuda.is_available():
            self.llm = self.llm.cuda()
            print("Models loaded on GPU")
        else:
            print("Models loaded on CPU")

    def load_faiss_index(self, index_path):
        print("Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        print("FAISS index loaded")

    def chunk_text(self, text, max_length=500):
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

    def search(self, query, k=5):
        """Search for relevant documents using FAISS."""
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        return [self.corpus[i] for i in indices[0]]

    def summarize_context(self, retrieved_docs, max_chunk_length=500):
        """Summarize context with chunking for long texts."""
        combined_context = " ".join(retrieved_docs)
        chunks = self.chunk_text(combined_context, max_chunk_length)
        
        summaries = []
        for chunk in chunks:
            try:
                summary = self.summarizer(chunk, max_length=50, min_length=30, do_sample=False)
                summaries.append(summary[0]["summary_text"])
            except Exception as e:
                print(f"Warning: Summarization failed for chunk: {e}")
                sentences = chunk.split('.')[:3]
                summaries.append('. '.join(sentences))
        
        return " ".join(summaries)

    def generate_response(self, context, question):
        """Generate response using the LLM."""
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        try:
            outputs = self.llm.generate(
                **inputs,
                max_length=1500,
                min_length=50,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error in response generation: {e}")
            return "Unable to generate response due to an error."

    def query(self, question):
        """Main query method for the RAG system."""
        try:
            print(f"\nProcessing query: {question}")
            
            # Search for relevant documents
            print("Searching relevant documents...")
            retrieved_docs = self.search(question)
            
            # Summarize the context
            print("Summarizing context...")
            summarized = self.summarize_context(retrieved_docs)
            
            # Generate response
            print("Generating response...")
            response = self.generate_response(summarized, question)
            
            return {
                "question": question,
                "answer": response,
                "context_summary": summarized
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "question": question,
                "answer": "An error occurred while processing your query.",
                "error": str(e)
            }

def main():
    # Initialize the RAG system
    rag = DrugRAGInference()
    
    # Interactive query loop
    print("\nDrug Discovery RAG System Ready!")
    print("Enter your questions (type 'exit' to quit)")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("Thank you for using the Drug Discovery RAG system!")
            break
            
        if not question:
            continue
            
        result = rag.query(question)
        
        print("\nAnswer:", result["answer"])
        print("\nContext Summary:", result["context_summary"])
        
        if "error" in result:
            print("\nError:", result["error"])

if __name__ == "__main__":
    main()
