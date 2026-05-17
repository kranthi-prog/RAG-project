import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
TOP_K = 4

PROMPT = PromptTemplate.from_template(
    "Use only the context below to answer the question. "
    "If the answer is not in the context, say 'I don't know'.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def build_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask(question: str):
    chain, retriever = build_chain()
    print(f"\nQuestion: {question}")
    print("-" * 50)
    answer = chain.invoke(question)
    print(f"Answer: {answer}")
    print("\nSources:")
    for doc in retriever.invoke(question):
        page = doc.metadata.get("page", "?")
        print(f"  [Page {page + 1}] {doc.page_content[:200]}...")


def main():
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        print("RAG Query CLI — type your question or 'quit' to exit\n")
        while True:
            question = input("Question: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if question:
                ask(question)


if __name__ == "__main__":
    main()
