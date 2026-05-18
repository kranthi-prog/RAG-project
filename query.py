import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"
TOP_K = 4
MEMORY_WINDOW = 6  # last 3 exchanges

PROMPT = PromptTemplate.from_template(
    "Use only the context below to answer the question. "
    "If the answer is not in the context, say 'I don't know'.\n\n"
    "Context:\n{context}\n\n"
    "Conversation History:\n{chat_history}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def format_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    recent = history[-MEMORY_WINDOW:]
    lines = []
    for role, content in recent:
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def load_components():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0)
    chain = PROMPT | llm | StrOutputParser()
    return retriever, chain


def ask(question: str, retriever, chain, history: list) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    chat_history = format_history(history)

    answer = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question,
    })

    print(f"\nAnswer: {answer}")
    print("\nSources:")
    for doc in docs:
        page = doc.metadata.get("page", "?")
        print(f"  [Page {page + 1}] {doc.page_content[:200]}...")

    return answer


def main():
    history = []

    if len(sys.argv) > 1:
        retriever, chain = load_components()
        question = " ".join(sys.argv[1:])
        print(f"\nQuestion: {question}")
        print("-" * 50)
        ask(question, retriever, chain, history)
    else:
        print("RAG Query CLI (with memory) — type your question or 'quit' to exit")
        print("The chatbot remembers your last 3 exchanges.\n")
        retriever, chain = load_components()

        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not question:
                continue

            print("-" * 50)
            answer = ask(question, retriever, chain, history)

            history.append(("Human", question))
            history.append(("Assistant", answer))


if __name__ == "__main__":
    main()
