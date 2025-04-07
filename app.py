import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from pdf_loader import load_and_split_pdfs

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("LLM_MODEL_PATH")

# Directories
OLD_PDF_DIR = "old_pdfs"
NEW_PDF_DIR = "new_pdfs"
CHROMA_DB_DIR = "chroma_db"

# Define system prompt
SYSTEM_PROMPT = """<|system|>
You are a helpful and intelligent legal assistant and have access to number of court cases. You would be provided some links to the court cases you need to identify the court cases accurately and then answer the questions.
"""

# Format prompt for LlamaCpp
def format_prompt(query: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}
<|user|>
[Context]: {context}
[Question]: {query}
<|assistant|>
"""

# Initialize the model
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=2048,
    temperature=0.1,
    max_tokens=512,
    top_k=40,
    n_threads=6,
    stop=["</s>"],
    verbose=False
)

# Load embedding model and vector DB
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding)

# Function to update Chroma DB with new PDFs
def update_chroma_if_new_pdfs():
    new_files = [f for f in os.listdir(NEW_PDF_DIR) if f.endswith(".pdf")]
    if not new_files:
        return

    st.info(f"🔄 Updating Chroma DB with {len(new_files)} new PDFs...")
    documents = load_and_split_pdfs(NEW_PDF_DIR)
    vectordb.add_documents(documents)
    vectordb.persist()

    for file in new_files:
        shutil.move(os.path.join(NEW_PDF_DIR, file), os.path.join(OLD_PDF_DIR, file))

    st.success("✅ Chroma DB updated and new PDFs moved to old_pdfs/")

# 🔁 Update at app startup
update_chroma_if_new_pdfs()

# Answer generation function
def get_answer(query: str) -> str:
    update_chroma_if_new_pdfs()  # Check for new PDFs before answering
    docs = vectordb.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = format_prompt(query, context)
    response = llm(prompt)
    return response.strip()

# Streamlit UI
st.title("📜 Court Case Q&A Assistant")
query = st.text_input("Ask your legal question:")

if query:
    with st.spinner("Thinking..."):
        result = get_answer(query)
        st.markdown(f"### 🧠 Answer:\n{result}")
