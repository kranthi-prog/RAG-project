import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"

PROMPT = PromptTemplate.from_template(
    "Use the following context to answer the question.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


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


def get_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM(model=OLLAMA_MODEL)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF RAG Chatbot")
st.caption(f"Powered by Ollama ({OLLAMA_MODEL}) + ChromaDB + sentence-transformers")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Step 1: Upload ────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.markdown("### Step 1: Upload your PDF")
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        st.markdown("### Step 2: Click the button below to index it")
        if st.button("Index PDF", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("Indexing PDF — this may take a minute..."):
                embeddings = get_embeddings()
                vectorstore, n_chunks = build_vectorstore(tmp_path, embeddings)
                st.session_state.vectorstore = vectorstore

            st.success(f"Done! Indexed {n_chunks} chunks from '{uploaded_file.name}'. You can now ask questions.")
            st.rerun()

# ── Step 2: Chat ──────────────────────────────────────────────────────────────
else:
    st.success("PDF is indexed and ready. Ask your questions below.")

    if st.button("Upload a different PDF"):
        del st.session_state.vectorstore
        st.session_state.messages = []
        st.rerun()

    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain, retriever = get_chain(st.session_state.vectorstore)
                answer = chain.invoke(prompt)
                sources = retriever.invoke(prompt)

            st.markdown(answer)

            with st.expander("View sources"):
                for doc in sources:
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Page {page + 1}:** {doc.page_content[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
