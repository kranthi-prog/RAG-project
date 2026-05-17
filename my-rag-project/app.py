import streamlit as st
from pathlib import Path
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vectorstore(pdf_path: str, embeddings):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    return vectorstore, len(chunks)


def get_qa_chain(vectorstore):
    llm = OllamaLLM(model=OLLAMA_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF RAG Chatbot")
st.caption(f"Powered by Ollama ({OLLAMA_MODEL}) + ChromaDB + sentence-transformers")

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if st.button("Index PDF", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("Indexing PDF..."):
                embeddings = get_embeddings()
                vectorstore, n_chunks = build_vectorstore(tmp_path, embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.messages = []

            st.success(f"Indexed {n_chunks} chunks from '{uploaded_file.name}'")

    if "vectorstore" in st.session_state:
        st.info("PDF indexed. Ask questions below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    if "vectorstore" not in st.session_state:
        st.warning("Please upload and index a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = get_qa_chain(st.session_state.vectorstore)
                result = chain.invoke({"query": prompt})
                answer = result["result"]
                sources = result["source_documents"]

            st.markdown(answer)

            with st.expander("Sources"):
                for doc in sources:
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Page {page + 1}:** {doc.page_content[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
