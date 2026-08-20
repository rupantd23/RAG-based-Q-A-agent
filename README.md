# RAG Q&A Agent with Gemini & LangChain

A lightweight, conversational Retrieval-Augmented Generation (RAG) agent built with Python. This project allows users to query private documents (like company policies) using Google's Gemini 1.5 Flash LLM, backed by local open-source vector embeddings for cost-effective semantic search.

## 🧠 Architecture & Tech Stack

This project implements modern LangChain Expression Language (LCEL) to create a history-aware retrieval pipeline. 

*   **LLM Generation:** Google Gemini 1.5 Flash (`langchain-google-genai`)
*   **Vectorization/Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local compute via `sentence-transformers`)
*   **Vector Database:** ChromaDB (Ephemeral local storage)
*   **Orchestration:** LangChain (LCEL)

## ✨ Key Features

*   **History-Aware Retrieval:** The agent reformulates user questions based on the chat history before querying the database, allowing for natural, multi-turn conversations.
*   **Anti-Hallucination Guardrails:** Strict system prompts prevent the model from answering questions outside the scope of the provided document context.
*   **Cost-Optimized:** Moves the heavy lifting of document embedding to your local CPU/GPU using open-source HuggingFace models, saving API costs.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed. You will also need a free Gemini API Key from [Google AI Studio](https://aistudio.google.com/).

### 2. Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/rupantd23/RAG-based-Q-A-agent.git](https://github.com/rupantd23/RAG-based-Q-A-agent.git)
cd RAG-based-Q-A-agent
pip install langchain langchain-google-genai langchain-huggingface langchain-chroma sentence-transformers wget
```

### 3. Environment Variables (Security Warning)

**Never hardcode your API key in the script.** Export your Gemini API key in your terminal before running the application:

Mac/Linux:

```bash
export GOOGLE_API_KEY="your_actual_api_key_here"
```
### 4. Running the Agent
**Execute the Python script to start the interactive terminal session:: 
```bash
python main.py
```

## 💡 How It Works

1.  **Ingestion:** The script downloads a sample text file (`companyPolicies.txt`), splits it into 1000-character chunks, and embeds them into a local ChromaDB instance.
2.  **Contextualization:** When you ask a question, the agent checks the chat history to see if it needs to rephrase your query for better database searching.
3.  **Retrieval & Generation:** The system fetches the most relevant document chunks and passes them to Gemini, which synthesizes a concise, fact-based answer.

## 🛠️ Future Improvements

*   [ ] Integrate `python-dotenv` for easier environment variable management.
*   [ ] Swap standard terminal input for a web UI using Streamlit or Gradio.
*   [ ] Implement a sliding window for chat history to prevent exceeding the context window limit in long conversations.
