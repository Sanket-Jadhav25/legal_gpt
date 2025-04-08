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
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH")
DEEPSEEK_MODEL_PATH = os.getenv("DEEPSEEK_MODEL_PATH")
MISTRAL_MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH")


# Directories
OLD_PDF_DIR = "old_pdfs"
NEW_PDF_DIR = "new_pdfs"
CHROMA_DB_DIR = "chroma_db"

# System prompt
SYSTEM_PROMPT = """<|system|>
You are a helpful and intelligent legal assistant and have access to number of court cases. You would be provided some links to the court cases you need to identify the court cases accurately and then answer the questions.
"""

# Prompt formatter
def format_prompt(query: str, context: str) -> str:
    return f"""{SYSTEM_PROMPT}
<|user|>
[Context]: {context}
[Question]: {query}
<|assistant|>
"""

# Model selection in Streamlit
st.title("📜 Court Case Q&A Assistant")
model_choice = st.selectbox("🧠 Select LLM Model", ["LLaMA", "DeepSeek", "Mistral"])
model_paths = {
    "LLaMA": LLAMA_MODEL_PATH,
    "DeepSeek": DEEPSEEK_MODEL_PATH,
    "Mistral": MISTRAL_MODEL_PATH
}
MODEL_PATH = model_paths[model_choice]
# Initialize the LLM
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

# Embeddings and Chroma vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding)

# Update Chroma DB with new PDFs
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

# 🔁 Initial update at app startup
update_chroma_if_new_pdfs()

# Answer generation
def get_answer(query: str) -> str:
    update_chroma_if_new_pdfs()  # Check for updates before each query
    docs = vectordb.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = format_prompt(query, context)
    return llm(prompt).strip()

# Streamlit input
query = st.text_input("Ask your legal question:")

if query:
    with st.spinner("Thinking..."):
        result = get_answer(query)
        st.markdown(f"### 🧠 Answer:\n{result}")
