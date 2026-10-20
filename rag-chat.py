import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from groq import RateLimitError
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # reads GROQ_API_KEY from the .env file

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

# Edge case: missing API key
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to a .env file next to app.py (see README).")
    st.stop()

# ---------- 1. Models (loaded once, then cached) ----------


@st.cache_resource
def load_embeddings():
    # Free, runs locally on your machine (Groq has no embedding model)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# The "strictly based on the document" rule lives in this system prompt
prompt = ChatPromptTemplate.from_messages(
   [
        (
            "system",
            """You answer questions about a PDF using ONLY the context given.
Rules:
1. Use only facts that are written in the context. Never use outside knowledge.
2. Do NOT add examples, definitions, background or explanations of your own,
   even if they would be helpful or you know them to be true.
3. Stay close to the wording of the context. Keep the answer short.
4. If the answer is not clearly in the context, reply exactly:
   "I couldn't find that in the document."
""",
        ),
        (
            "human",
            """Context:
{context}
 
Question: {question}
 
Answer using only the context above, with nothing added.""",
        ),
    ]
)

# ---------- 2. Build the vector database from the PDF ----------


def build_vector_db(pdf_file):
    # Save the upload to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

    pages = PyPDFLoader(pdf_path).load()  # extract text (one item per page)
    os.remove(pdf_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)  # split text into small chunks

    # Edge case: PDF has no readable text (e.g. scanned images)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        return None

    # Embed the chunks and store them in ChromaDB (kept in memory)
    return Chroma.from_documents(chunks, load_embeddings())


# ---------- 3. Streamlit UI ----------

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is None:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

# Only rebuild the database when a new file is uploaded
if st.session_state.get("file_name") != uploaded_file.name:
    try:
        with st.spinner("Reading and indexing your PDF..."):
            db = build_vector_db(uploaded_file)
    except Exception as e:  # corrupted or password-protected PDF
        st.error(f"Could not read this PDF: {e}")
        st.stop()

    if db is None:
        st.error("No text found in this PDF. It may be a scanned document.")
        st.stop()

    st.session_state.db = db
    st.session_state.file_name = uploaded_file.name
    st.session_state.messages = []  # clear old chat

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- 4. Ask a question ----------

question = st.chat_input("Ask something about your PDF")

if question:
    question = question.strip()

    # Edge case: empty input
    if not question:
        st.warning("Please type a question.")
        st.stop()

    with st.chat_message("user"):
        st.write(question)

    # Find the 4 most relevant chunks
    docs = st.session_state.db.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)

    with st.chat_message("assistant"):
        try:
            answer = (prompt | llm).invoke(
                {"context": context, "question": question}
            ).content
        except RateLimitError:
            # Edge case: Groq rate limit
            st.error("Groq rate limit reached. Wait a minute and try again.")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong calling the model: {e}")
            st.stop()

        st.write(answer)
        with st.expander("Sources"):
            for doc in docs:
                st.caption(f"Page {doc.metadata['page'] + 1}")
                st.write(doc.page_content)

    # Save the exchange only after it succeeded
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})