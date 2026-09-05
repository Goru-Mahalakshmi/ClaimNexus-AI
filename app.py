import os
import json
import numpy as np
from flask import Flask, request, render_template_string
from google import genai

app = Flask(__name__)

# Initialize the Google GenAI client using the environment variable
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# -----------------------------------------------------------------------------
# Local Knowledge Base & Document Store (RAG Grounding Material)
# -----------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    {
        "id": "clause_1_1",
        "title": "Clause 1.1 (Coverage)",
        "content": "The policy covers accidental damage from collisions, provided the fundamental facts of how the collision occurred are consistent across all evidentiary records."
    },
    {
        "id": "clause_2_1",
        "title": "Clause 2.1 (Required Documents)",
        "content": "Mandatory submission documents include the Claim Form, Customer Incident Description, and a certified Repair Estimate."
    },
    {
        "id": "clause_2_2",
        "title": "Clause 2.2 (FIR Requirement)",
        "content": "An official First Information Report (FIR) or police report must be provided when third-party involvement or moving vehicle collisions are claimed. Its contents must align with the customer's primary accident narrative."
    }
]

# Simple Local Vector Store using gemini-embedding-001
class LocalVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.embeddings = []
        self._build_index()

    def _build_index(self):
        print("Initializing FAISS-style local vector store with gemini-embedding-001...")
        for doc in self.documents:
            text = f"{doc['title']}: {doc['content']}"
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text
                )
                vector = response.embeddings[0].values
                self.embeddings.append(np.array(vector, dtype=np.float32))
            except Exception as e:
                print(f"Embedding-001 execution note: {e}")
                self.embeddings.append(np.zeros(768, dtype=np.float32))

    def similarity_search(self, query, top_k=2):
        try:
            res = client.models.embed_content(
                model="gemini-embedding-001",
                contents=query
            )
            q_vector = np.array(res.embeddings[0].values, dtype=np.float32)
            
            scores = []
            for idx, emb in enumerate(self.embeddings):
                dot_prod = np.dot(q_vector, emb)
                norm = np.linalg.norm(q_vector) * np.linalg.norm(emb)
                score = dot_prod / norm if norm > 0 else 0.0
                scores.append((score, idx))
            
            scores.sort(key=lambda x: x[0], reverse=True)
            return [self.documents[idx] for _, idx in scores[:top_k]]
        except Exception as e:
            print(f"Search error: {e}")
            return self.documents[:top_k]

vector_store = LocalVectorStore(KNOWLEDGE_BASE)

# -----------------------------------------------------------------------------
# Frontend Template
# -----------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ClaimNexus AI - Insurance Evidence Review</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f4f6f9; margin: 0; padding: 40px; color: #333; }
        .container { max-width: 800px; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin: auto; }
        h1 { color: #1a365d; font-size: 24px; margin-bottom: 10px; }
        p { color: #555; }
        textarea { width: 100%; height: 140px; padding: 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-family: monospace; font-size: 14px; margin-top: 10px; box-sizing: border-box; }
        button { background: #2563eb; color: #fff; border: none; padding: 12px 24px; font-size: 16px; font-weight: 600; border-radius: 6px; cursor: pointer; margin-top: 15px; transition: background 0.2s; }
        button:hover { background: #1d4ed8; }
        .result-box { margin-top: 25px; padding: 20px; background: #f8fafc; border-left: 4px solid #2563eb; border-radius: 4px; white-space: pre-wrap; line-height: 1.5; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ClaimNexus AI — Evidence Review Assistant (PS02)</h1>
        <p>Paste your JSON claim payload below to test deterministic validation and RAG evidence grounding:</p>
        <form method="POST" action="/api/review">
            <textarea name="payload">{{ payload }}</textarea><br>
            <button type="submit">Review Claim Evidence</button>
        </form>
        {% if result %}
        <div class="result-box"><strong>RESULT:</strong><br>{{ result }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    default_payload = json.dumps({
        "claim_form": "Claim ID: 90210, Date of Incident: 2026-08-15, Insured Value: $15,000",
        "customer_description": "I was driving on Main St and a deer jumped out. I swerved and hit a tree.",
        "repair_estimate": "Front bumper replacement, radiator repair. Total: $3,200",
        "fir": "Police report states driver was rear-ended by an unknown vehicle at a stoplight."
    }, indent=2)
    return render_template_string(HTML_TEMPLATE, payload=default_payload, result=None)

@app.route("/api/review", methods=["POST"])
def review_claim():
    payload_str = request.form.get("payload") or request.data.decode("utf-8")
    
    relevant_docs = vector_store.similarity_search(payload_str, top_k=2)
    rag_context = "\n".join([f"- {doc['title']}: {doc['content']}" for doc in relevant_docs])

    prompt = f"""
    You are an expert Insurance Evidence Review AI assistant. 
    Analyze the following insurance claim JSON payload against the provided policy clauses (RAG Context).
    
    Policy Clauses Context:
    {rag_context}
    
    Claim Payload:
    {payload_str}
    
    Provide your response in clear, structured sections:
    - RECOMMENDATION: (e.g. Escalate / Approve / Reject)
    - DAMAGE VECTOR CHECK: Analysis of physical impact match between description, estimates, and FIR.
    - KEY FINDINGS & CONTRADICTIONS: Bullet points of exact contradictions found.
    - POLICY ALIGNMENT: Direct clause citations mapping to findings.
    """

    try:
        # Updated to use the correct model identifier: gemini-3.5-flash-lite
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
        )
        result_text = response.text
    except Exception as e:
        result_text = f"Error during model generation: {str(e)}"

    return render_template_string(HTML_TEMPLATE, payload=payload_str, result=result_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)