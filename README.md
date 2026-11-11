🚀 *RAG Chatbot with Self-Correcting Evaluation Pipeline*

A document question-answering system built with *LangGraph* that doesn't just retrieve and generate — it evaluates its own answers for faithfulness, automatically retries when confidence is low, and ships with a full RAGAS-based evaluation harness to quantify how well it actually performs.

Built as a portfolio project for AI/GenAI Engineer roles, with a focus on the parts most RAG demos skip: automated evaluation, self-correction, and honest reporting of results.

────────────────────────────

💡 *WHY THIS PROJECT IS DIFFERENT*

Most "RAG chatbot" portfolio projects stop at retrieval + generation. This one adds the layer that answers the question every RAG interview eventually asks — *"how do you know it's not hallucinating?"*

🔹 *Self-correcting retrieval loop* — Every answer is scored for faithfulness against its retrieved context before being finalized. If the score falls below threshold, the system rewrites the query and retries (capped at 2 attempts) instead of silently returning a weak answer.
🔹 *Formal evaluation harness* — A 15-question golden dataset scored on faithfulness, answer relevancy, and context precision using RAGAS, with real, reproducible numbers.
🔹 *Production-shaped chatbot UI* — Multi-session chat with persistent history, per-session document upload, and RAG-only chat gating (no answers without grounding context).

────────────────────────────

📊 *EVALUATION RESULTS*

Run against a 15-question golden dataset (fictional 10-K style document, `dataset.txt`), scored with RAGAS using an LLM-as-judge:

• *Faithfulness:* 1.000
• *Answer Relevancy:* 0.826
• *Context Precision:* 0.867
• *Queries triggering retry:* 0 / 15
• *Total wall-clock time:* ~17.4 minutes

📌 *The self-correction retry loop is a core part of the pipeline's design* — every answer is gated on a faithfulness check before being finalized. In this run, all 15 golden-dataset queries passed the gate on the first attempt, so no retries were needed. The retry mechanism was separately validated with out-of-distribution queries (e.g., asking about information not present in the source document), where it correctly triggers a query rewrite and, on the second attempt, produces a grounded refusal instead of a hallucinated answer.

Full per-query results and the evaluation script are in `run.py` and `golden_dataset.py`.

────────────────────────────

⚙️ *ARCHITECTURE*

query ──► [ retrieve ] (FAISS similarity search)
                 │
                 ▼
          [ generate ] (LLM answers from retrieved context)
                 │
                 ▼
          [ evaluate ] (RAGAS faithfulness gate-check)
                 │
        Score < threshold?
          ├── YES ──► [ rewrite_query ] ──► (back to retrieve, max 2 retries)
          └── NO  ──► Done

Built as a `LangGraph` `StateGraph` with a conditional edge controlling the retry loop. State (query, retrieved docs, answer, eval scores, retry count, per-node latency) is threaded through every node as a typed dict.

🛠️ *Two-Tier Evaluation Design:*
• *In-graph gate-check (`evaluate_node`):* Scores faithfulness only on every attempt to keep the retry loop fast.
• *Full batched report (`run.py`):* Computes faithfulness + answer relevancy + context precision across the entire golden dataset. Splitting these was a deliberate latency optimization.

────────────────────────────

💻 *TECH STACK*

• *Orchestration:* LangGraph (`StateGraph`, conditional edges)
• *LLM (Gen + Judge):* Groq (`openai/gpt-oss-20b`)
• *Embeddings:* Google Generative AI (`gemini-embedding-001`)
• *Vector Store:* FAISS
• *Evaluation:* RAGAS (faithfulness, answer relevancy, context precision)
• *UI:* Streamlit
• *Document Parsing:* PyPDFLoader / TextLoader (PDF, TXT, MD)

📌 *Why Groq for both generation and judging:* Gemini was tried first as the RAGAS judge model, but its structured-output formatting for RAGAS internal prompts didn't reliably match parser expectations (`RagasOutputParserException`). Groq's OpenAI-compatible output resolved this consistently. Google is still used for embeddings.

────────────────────────────

✨ *FEATURES*

• *Multi-session chat:* Create new chats, switch between past sessions, delete old ones. History persists to disk (JSON).
• *Per-session RAG:* Upload PDF/TXT to ground answers; session documents and vector indexes are completely isolated.
• *RAG-only gating:* Refuses ungrounded questions until a document is attached.
• *Transparent scoring:* Every chat response shows its faithfulness score and retry count inline.

────────────────────────────

📁 *PROJECT STRUCTURE*

code/
├── app.py              # Streamlit chatbot UI
├── backend.py          # LangGraph pipeline, RAGAS eval, session management
├── run.py              # Golden dataset evaluation runner
├── golden_dataset.py   # 15 Q&A pairs with ground truth, used for eval
├── dataset.txt         # Source document for the golden dataset
└── requirements.txt

────────────────────────────

🚀 *SETUP & RUN*

1️⃣ *Install dependencies:*
`pip install -r requirements.txt`

2️⃣ *Create `.env` file:*
`GROQ_API_KEY=your_key_here`
`GOOGLE_API_KEY=your_key_here`

3️⃣ *Run Chatbot:*
`streamlit run app.py`

4️⃣ *Run Evaluation Suite:*
`python run.py`

────────────────────────────

⚠️ *KNOWN LIMITATIONS*

• *Vectorstore concurrency:* Swaps a module-level `_vectorstore` reference per session call. Fine for single-user; needs refactoring to per-request state for multi-user access.
• *Session storage:* Uses flat JSON files and local FAISS folders. Needs Postgres + managed vector DB for production.
• *Free-tier limits:* Handled Hugging Face credit limits, Gemini 2.5 Flash deprecation, and RAGAS version bugs during build.