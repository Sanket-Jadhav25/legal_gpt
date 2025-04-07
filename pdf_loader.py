from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_split_pdfs(pdf_dir):
    docs = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)
