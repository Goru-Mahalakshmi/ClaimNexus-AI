\# ClaimNexus AI



\## Multimodal Claims Evidence Review \& Investigation Workbench



\*\*TRACK\_ID:\*\* PS02



ClaimNexus AI is an AI-powered motor insurance claims review system. It uses RAG and Google Gemini to analyze claims, identify contradictions, check policy alignment, calculate risk, and provide investigator recommendations.



\## Features



\- 🔍 Claim evidence analysis

\- ⚠️ Contradiction detection

\- 📄 Policy verification using RAG

\- 🤖 Google Gemini multimodal AI

\- 📊 Risk scoring

\- 👨‍💼 Investigator recommendations

\- ✅ Deterministic validation



\## Project Structure



```text

ClaimNexus-AI/

│

├── data/

│   ├── claim\_contradiction.json

│   ├── claim\_happy.json

│   ├── claim\_missing.json

│   └── policy\_master.txt

│

├── frontend/

│   └── templates/

│       └── index.html

│

├── src/

│   └── rag\_engine.py

│

├── .env.example

├── app.py

├── README.md

└── requirements.txt

Installation

1\. Install dependencies

pip install -r requirements.txt

2\. Configure API Key

Create a .env file:

GEMINI\_API\_KEY=your\_api\_key\_here

3\. Run the application

python app.py

Open the URL shown in the terminal.

How It Works

Claim + Evidence

&#x20;      ↓

Validation

&#x20;      ↓

RAG Policy Retrieval

&#x20;      ↓

Gemini AI Analysis

&#x20;      ↓

Contradiction Detection

&#x20;      ↓

Policy Check

&#x20;      ↓

Risk Score

&#x20;      ↓

Investigator Recommendation

Technology Stack

Python

Google Gemini

RAG

HTML

JSON

Policy Knowledge Base

Disclaimer

ClaimNexus AI provides AI-assisted recommendations. Final insurance claim decisions should always be made by an authorized human investigator.

