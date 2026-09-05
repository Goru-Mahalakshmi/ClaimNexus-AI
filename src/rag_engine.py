import numpy as np
import faiss

class LocalVectorStore:
    def __init__(self, documents, client):
        self.documents = documents
        self.client = client
        self.dimension = 768  # gemini-embedding-001 outputs 768 dimensions
        # Use Inner Product index. When vectors are L2 normalized, this equates to Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dimension) 
        self._build_index()

    def _build_index(self):
        print("Initializing FAISS local vector store with gemini-embedding-001...")
        for doc in self.documents:
            text = f"{doc['title']}: {doc['content']}"
            try:
                response = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text
                )
                vector = np.array(response.embeddings[0].values, dtype=np.float32).reshape(1, -1)
                
                # Normalize vector for accurate cosine similarity in FAISS
                faiss.normalize_L2(vector)
                self.index.add(vector)
            except Exception as e:
                print(f"Embedding generation note: {e}")
                # Add a zero vector to maintain index alignment if an embedding fails
                dummy_vector = np.zeros((1, self.dimension), dtype=np.float32)
                self.index.add(dummy_vector)

    def similarity_search(self, query, top_k=2):
        try:
            res = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=query
            )
            q_vector = np.array(res.embeddings[0].values, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(q_vector)
            
            # FAISS search returns distances (scores) and indices of the nearest neighbors
            distances, indices = self.index.search(q_vector, top_k)
            
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.documents):
                    results.append(self.documents[idx])
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return self.documents[:top_k]