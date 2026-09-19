import os
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


load_dotenv()

app = Flask(__name__)

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

qa_chain = None
startup_error = None

TRANSLITERATED_QUERY_HINTS = (
    "haga",
    "paykhana",
    "pakhana",
    "potty",
)

BANGLA_QUERY_HINTS = {
    "\u09ac\u09cd\u09b0\u09a3": "acne",
    "\u09ae\u09c1\u0996": "face skin",
    "\u09b9\u09c3\u09a6\u09b0\u09cb\u0997": "heart disease",
    "\u0995\u09cb\u09b7\u09cd\u09a0\u0995\u09be\u09a0\09bf\u09a8\u09cd\u09af": "constipation difficulty passing stool",
    "\u09aa\u09be\u09df\u0996\u09be\u09a8\u09be": "bowel movement constipation",
}


def expand_query_for_retrieval(message: str) -> str:
    normalized = message.lower()
    if any(term in normalized for term in TRANSLITERATED_QUERY_HINTS):
        return (
            f"{message}\n"
            "Medical meaning: constipation, difficulty passing stool, or no bowel movement. "
            "Find practical guidance and warning signs from the medical book."
        )
    matched_topics = [hint for term, hint in BANGLA_QUERY_HINTS.items() if term in message]
    if matched_topics:
        return (
            f"{message}\nMedical meaning: {', '.join(matched_topics)}. "
            "Find practical guidance and warning signs from the medical book."
        )
    return message


def response_language(message: str) -> str:
    if any(term in message.lower() for term in TRANSLITERATED_QUERY_HINTS):
        return "Bangla"
    if any("\u0980" <= character <= "\u09ff" for character in message):
        return "Bangla"
    return "English"


class UniqueDocumentRetriever(BaseRetriever):
    vector_store: Any
    search_k: int = 32
    result_k: int = 8

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None):
        documents = self.vector_store.similarity_search(query, k=self.search_k)
        unique_documents = []
        seen_content = set()
        for document in documents:
            content = document.page_content.strip()
            if not content or content in seen_content:
                continue
            seen_content.add(content)
            unique_documents.append(document)
            if len(unique_documents) >= self.result_k:
                break
        return unique_documents


def required_environment_variables():
    missing = []
    if not os.getenv("PINECONE_API_KEY"):
        missing.append("PINECONE_API_KEY")

    if LLM_PROVIDER == "groq":
        if not os.getenv("GROQ_API_KEY"):
            missing.append("GROQ_API_KEY")
    elif LLM_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
    else:
        missing.append("LLM_PROVIDER must be 'groq' or 'openai'")

    return missing


def build_llm():
    if LLM_PROVIDER == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: install langchain-groq to use Groq."
            ) from exc

        return ChatGroq(model=GROQ_MODEL, temperature=0.2)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)


def build_qa_chain():
    missing = required_environment_variables()
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )

    embeddings = download_hugging_face_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )
    retriever = UniqueDocumentRetriever(vector_store=docsearch)

    prompt = PromptTemplate(
        template=system_prompt + "\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"],
    )
    llm = build_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def get_qa_chain():
    global qa_chain, startup_error

    if qa_chain is not None:
        return qa_chain
    try:
        qa_chain = build_qa_chain()
        startup_error = None
    except Exception as exc:
        startup_error = str(exc)
        raise
    return qa_chain


@app.route("/")
def index():
    missing = required_environment_variables()
    current_error = None
    if missing:
        current_error = "Missing required environment variable(s): " + ", ".join(missing)
    elif startup_error and "Missing required environment variable" not in startup_error:
        current_error = startup_error

    return render_template("chat.html", startup_error=current_error)


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "index": PINECONE_INDEX_NAME,
            "llm_provider": LLM_PROVIDER,
        }
    )


@app.route("/get", methods=["POST"])
def chat():
    message = request.form.get("msg")
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        message = payload.get("msg", message)

    if not message or not message.strip():
        return jsonify({"error": "Please send a non-empty message."}), 400

    normalized_message = message.strip().lower()
    if normalized_message in {"hello", "hi", "hey", "good morning", "good afternoon"}:
        return jsonify(
            {
                "answer": "Hello! Ask me a medical question, and I will answer using the indexed medical book."
            }
        )

    try:
        query = expand_query_for_retrieval(message.strip())
        if response_language(message.strip()) == "Bangla":
            query += "\nAnswer in Bangla."
        result = get_qa_chain().invoke({"query": query})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"answer": result.get("result", "")})


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=debug)
