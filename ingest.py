import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCUMENTS_DIR = "./documents"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def load_documents():
    docs = []
    for filename in os.listdir(DOCUMENTS_DIR):
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        else:
            continue
        docs.extend(loader.load())
        print(f"Loaded: {filename}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def main():
    print("Step 1/4: Loading documents...")
    docs = load_documents()
    if not docs:
        print("No documents found in ./documents. Add PDF, TXT or DOCX files and re-run.")
        return
    print(f"  Loaded {len(docs)} pages from {DOCUMENTS_DIR}")

    print("Step 2/4: Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"  Created {len(chunks)} chunks")

    print("Step 3/4: Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("Step 4/4: Storing in ChromaDB...")
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print(f"  Done! Vector store saved to {CHROMA_DIR}")


if __name__ == "__main__":
    main()
