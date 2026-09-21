import os

import streamlit as st
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from groq import Groq

import fitz  # PyMuPDF
from docx import Document


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📚",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📚 HR Policy Assistant")
st.write(
    "Upload HR policy documents and ask questions about them. "
    "The assistant answers using only the uploaded documents."
)


# ---------------------------------------------------------
# LOAD SENTENCE TRANSFORMER MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    """
    Load the Sentence Transformer model.

    The model converts text into numerical vectors
    called embeddings.
    """

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


embedding_model = load_embedding_model()


# ---------------------------------------------------------
# GET GROQ API KEY
# ---------------------------------------------------------

def get_groq_client():
    """
    Get the Groq API key from Streamlit secrets.
    """

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error(
            "GROQ_API_KEY was not found. "
            "Please add it to Streamlit Cloud Secrets."
        )
        st.stop()

    return Groq(api_key=api_key)


groq_client = get_groq_client()


# ---------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------

def extract_text_from_pdf(file):
    """
    Extract text from a PDF file.
    """

    text = ""

    pdf = fitz.open(stream=file.read(), filetype="pdf")

    for page in pdf:
        text += page.get_text()

    pdf.close()

    return text


def extract_text_from_docx(file):
    """
    Extract text from a DOCX file.
    """

    document = Document(file)

    text = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text.append(paragraph.text)

    return "\n".join(text)


def extract_text_from_txt(file):
    """
    Extract text from a TXT file.
    """

    return file.read().decode("utf-8", errors="ignore")


def extract_text(file):
    """
    Select the correct text extraction method
    according to the uploaded file type.
    """

    file_name = file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file)

    elif file_name.endswith(".docx"):
        return extract_text_from_docx(file)

    elif file_name.endswith(".txt"):
        return extract_text_from_txt(file)

    else:
        return ""


# ---------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------

def split_text(text, chunk_size=800, overlap=150):
    """
    Split a large document into smaller pieces.

    chunk_size:
        Approximate number of characters in each chunk.

    overlap:
        Number of characters repeated between chunks.
        This helps preserve context.
    """

    text = text.replace("\x00", " ")

    words = text.split()

    chunks = []

    current_chunk = []

    current_length = 0

    for word in words:

        current_chunk.append(word)

        current_length += len(word) + 1

        if current_length >= chunk_size:

            chunk = " ".join(current_chunk)

            chunks.append(chunk)

            overlap_words = []

            overlap_length = 0

            for previous_word in reversed(current_chunk):

                overlap_words.insert(0, previous_word)

                overlap_length += len(previous_word) + 1

                if overlap_length >= overlap:
                    break

            current_chunk = overlap_words

            current_length = overlap_length

    if current_chunk:

        chunks.append(" ".join(current_chunk))

    return chunks


# ---------------------------------------------------------
# CREATE FAISS INDEX
# ---------------------------------------------------------

def create_faiss_index(chunks):
    """
    Convert chunks into embeddings and store them
    inside a FAISS index.
    """

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    # Normalize vectors so inner product works like
    # cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ---------------------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# ---------------------------------------------------------

def retrieve_relevant_chunks(question, index, chunks, top_k=5):
    """
    Search FAISS for the chunks most similar
    to the user's question.
    """

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype("float32")

    faiss.normalize_L2(question_embedding)

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_number in zip(scores[0], indices[0]):

        if index_number >= 0:

            results.append(
                {
                    "text": chunks[index_number]["text"],
                    "source": chunks[index_number]["source"],
                    "score": float(score)
                }
            )

    return results


# ---------------------------------------------------------
# ASK GROQ
# ---------------------------------------------------------

def generate_answer(question, retrieved_chunks):
    """
    Send the user's question and retrieved policy
    information to Groq.
    """

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"Source: {item['source']}\n"
            f"Policy text:\n{item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are an HR Policy Assistant.

Answer the user's question ONLY using the HR policy
information provided below.

Do not use outside knowledge.

If the answer cannot be found in the provided policy
information, clearly say:

"I could not find this information in the uploaded
HR policies."

Do not invent or guess information.

User question:
{question}

Uploaded HR policy information:
{context}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful HR policy assistant. "
                    "Use only the provided policy context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload HR policy documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)


# ---------------------------------------------------------
# PROCESS DOCUMENTS
# ---------------------------------------------------------

if uploaded_files:

    all_chunks = []

    st.subheader("Uploaded Documents")

    for file in uploaded_files:

        st.write(f"📄 {file.name}")

        text = extract_text(file)

        if not text.strip():

            st.warning(
                f"No readable text was found in {file.name}."
            )

            continue

        document_chunks = split_text(text)

        for chunk in document_chunks:

            all_chunks.append(
                {
                    "text": chunk,
                    "source": file.name
                }
            )

    if all_chunks:

        st.success(
            f"Created {len(all_chunks)} searchable policy sections."
        )

        # Create FAISS index
        index = create_faiss_index(all_chunks)

        st.session_state["index"] = index
        st.session_state["chunks"] = all_chunks

    else:

        st.error(
            "No readable text was found in the uploaded documents."
        )


# ---------------------------------------------------------
# QUESTION SECTION
# ---------------------------------------------------------

st.divider()

st.subheader("Ask a Question")

question = st.text_input(
    "Enter your HR policy question:",
    placeholder="Example: How many annual leaves are allowed?"
)


# ---------------------------------------------------------
# ANSWER
# ---------------------------------------------------------

if st.button("🔎 Ask Question"):

    if not uploaded_files:

        st.warning(
            "Please upload at least one HR policy document first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif "index" not in st.session_state:

        st.warning(
            "Please upload readable HR policy documents first."
        )

    else:

        with st.spinner("Searching HR policies..."):

            retrieved_chunks = retrieve_relevant_chunks(
                question,
                st.session_state["index"],
                st.session_state["chunks"],
                top_k=5
            )

        with st.spinner("Generating answer..."):

            answer = generate_answer(
                question,
                retrieved_chunks
            )

        st.subheader("Answer")

        st.write(answer)

        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        st.subheader("📌 Retrieved Sources")

        for number, item in enumerate(
            retrieved_chunks,
            start=1
        ):

            with st.expander(
                f"{number}. {item['source']}"
            ):

                st.write(item["text"])

                st.caption(
                    f"Similarity score: {item['score']:.3f}"
                )
