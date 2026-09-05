import os
import json
from flask import Flask, render_template, request, jsonify
from google import genai

from src.rag_engine import LocalVectorStore

# Point Flask to the custom frontend/templates directory
app = Flask(__name__, template_folder='frontend/templates')

try:
    client = genai.Client()
except Exception as e:
    print(f"Failed to initialize GenAI client: {e}")
    client = None

vector_store = None

def load_policies_into_rag():
    global vector_store
    documents = []
    try:
        with open("data/policy_master.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if ":" in line:
                    title, content = line.split(":", 1)
                    documents.append({"title": title.strip(), "content": content.strip()})
        
        if client:
            vector_store = LocalVectorStore(documents, client)
            print("FAISS Vector Store successfully initialized.")
    except Exception as e:
        print(f"Error loading policies: {e}")

load_policies_into_rag()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_claim():
    req_data = request.get_json()
    if not req_data:
        return jsonify({"error": "No payload provided"}), 400

    claim_id = req_data.get("claim_id", "Unknown")
    customer_desc = req_data.get("customer_description", "")
    fir_details = req_data.get("fir", "")
    
    # 1. FAISS RAG Retrieval Phase
    search_query = f"Collision, hit-and-run, documentation requirements, impact vectors for: {customer_desc}. Police report states: {fir_details}"
    
    retrieved_policies = []
    if vector_store:
        retrieved_policies = vector_store.similarity_search(search_query, top_k=3)
    
    context_text = "\n".join([f"- {doc['title']}: {doc['content']}" for doc in retrieved_policies])

    # 2. LLM Reasoning Phase using Gemini 3.5 Flash Lite
    prompt = f"""
    You are an expert insurance claims fraud investigator. Evaluate the following claim data strictly against the retrieved policy clauses.
    Do not invent information. If information is missing, explicitly state it.
    
    RELEVANT POLICY CLAUSES:
    {context_text}
    
    CLAIM DATA:
    {json.dumps(req_data, indent=2)}
    
    Provide your response formatted exactly as HTML snippet to inject into a dashboard:
    1. A short <p><strong>Analysis:</strong> ...</p> summarizing the findings.
    2. A <ul> with classes 'list-disc pl-5 space-y-2 text-xs text-slate-300' containing:
       - <li><strong>Claim ID:</strong> [ID]</li>
       - <li><strong>Evidence Check:</strong> [Explain contradictions, alignments, or missing docs based on Policy]</li>
       - <li><strong>Recommendation:</strong> <span class="[color-class] font-bold">[APPROVE / REJECT / REQUEST_INFO]</span></li>
       
    Use text-rose-400 for REJECT, text-emerald-400 for APPROVE, text-amber-400 for REQUEST_INFO.
    Also, prefix your response with a Risk Indicator badge rating (e.g., RISK_SCORE: 85).
    """

    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt
        )
        output_text = response.text
        
        risk_score = 50
        html_output = output_text
        if "RISK_SCORE:" in output_text:
            parts = output_text.split("RISK_SCORE:")
            try:
                risk_score_str = parts[1].split()[0][:2]
                risk_score = int(risk_score_str)
            except:
                pass
            html_output = parts[1].split("\n", 1)[1] 
            
        return jsonify({
            "html": html_output.replace("```html", "").replace("```", "").strip(),
            "risk_score": risk_score
        })
        
    except Exception as e:
        return jsonify({"html": f"<p class='text-rose-400'>Analysis Failed: {str(e)}</p>", "risk_score": 100})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)