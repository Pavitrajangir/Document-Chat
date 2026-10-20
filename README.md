# 📄 Chat with your PDF

A simple RAG (Retrieval-Augmented Generation) app. Upload a PDF, ask questions, and get answers based **only** on that document.

**Tech stack:** Python, LangChain, ChromaDB, Streamlit, Groq (`openai/gpt-oss-120b`), HuggingFace embeddings

## How it works

1. **Extract**: `pypdf` reads the text from every page
2. **Split**: `RecursiveCharacterTextSplitter` cuts it into 1000-character chunks (200 overlap)
3. **Embed + store**: chunks are embedded locally (`all-MiniLM-L6-v2`) and saved in ChromaDB
4. **Retrieve**: for each question, the 4 most similar chunks are fetched
5. **Answer**: only those chunks + the question go to the Groq LLM, which is told to answer strictly from them

## Setup

### 1. Install
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

### 2. Add your API key (`.env`)
Get a free key at https://console.groq.com/keys, then create a file named `.env` next to `app.py`:

```
GROQ_API_KEY=your_actual_key_here
```
(`.env.example` shows the format. Never commit your real `.env` to GitHub.)

### 3. Run
```bash
streamlit run app.py
```
The first run downloads the embedding model (~90 MB) once.

## Edge cases handled

| Situation | What the app does |
|---|---|
| Missing API key | Shows a clear error and stops |
| Answer not in the PDF (hallucination) | The system prompt tells the LLM to use only the context, otherwise reply "I couldn't find that in the document." |
| Groq rate limit / API error | Shows the error message instead of crashing |
| Scanned PDF (no text) | Shows an error instead of crashing |

## Settings you can tweak (in `app.py`)

- The system prompt (`prompt` in `app.py`): the rules the LLM must follow
- `chunk_size` / `chunk_overlap`: how the PDF is split
- `k=4`: how many chunks are retrieved per question

## Troubleshooting

- **"GROQ_API_KEY not found"**: make sure the file is named exactly `.env` (not `.env.txt`) and sits next to `app.py`
- **Model still answers from outside the PDF**: make the system prompt rules stricter, or lower `temperature` (already `0`)
- **Scanned PDF**: this app reads text only; it has no OCR