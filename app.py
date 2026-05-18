import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
MEMORY_WINDOW = 6  # number of recent messages to keep as context (3 exchanges)

PROMPT = PromptTemplate.from_template(
    "Use only the context below to answer the question. "
    "If the answer is not in the context, say 'I don't know'.\n\n"
    "Context:\n{context}\n\n"
    "Conversation History:\n{chat_history}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vectorstore(pdf_path: str, embeddings):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    return vectorstore, len(chunks)


def format_history(messages: list) -> str:
    if not messages:
        return "No previous conversation."
    recent = messages[-MEMORY_WINDOW:]
    lines = []
    for msg in recent:
        role = "Human" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def get_answer(vectorstore, question: str, chat_history: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0)

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question,
    })
    return answer, docs


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF RAG Chatbot")
st.caption(f"Powered by Ollama ({OLLAMA_MODEL}) + ChromaDB + sentence-transformers | Memory: last {MEMORY_WINDOW // 2} exchanges")

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
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success("PDF is indexed and ready. Ask your questions below.")
    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

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
                history = format_history(st.session_state.messages[:-1])
                answer, sources = get_answer(
                    st.session_state.vectorstore, prompt, history
                )

            st.markdown(answer)

            with st.expander("View sources"):
                for doc in sources:
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Page {page + 1}:** {doc.page_content[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
