from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from pdf_loader import load_and_split_pdfs
import os
import shutil

def move_to_old_pdfs(src_dir="new_pdfs", dest_dir="old_pdfs"):
    for filename in os.listdir(src_dir):
        if filename.endswith(".pdf"):
            shutil.move(os.path.join(src_dir, filename), os.path.join(dest_dir, filename))

def create_or_update_vector_store(new_pdf_dir="new_pdfs", old_pdf_dir="old_pdfs", persist_dir="chroma_db"):
    # Load and split new PDFs
    documents = load_and_split_pdfs(new_pdf_dir)

    if documents:
        # Load embeddings and ChromaDB
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)

        # Add new documents
        vectordb.add_documents(documents)
        vectordb.persist()

        # Move processed PDFs to old_pdfs
        move_to_old_pdfs(new_pdf_dir, old_pdf_dir)
    else:
        print("No new PDFs found.")
