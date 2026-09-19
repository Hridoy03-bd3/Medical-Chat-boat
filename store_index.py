import os
import time

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split


load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_DIMENSION = 384


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def ensure_index(pc: Pinecone, index_name: str) -> None:
    existing_indexes = {index.name for index in pc.list_indexes()}
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)


def main() -> None:
    api_key = require_env("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    ensure_index(pc, PINECONE_INDEX_NAME)

    documents = load_pdf_file(DATA_PATH)
    chunks = text_split(documents)
    embeddings = download_hugging_face_embeddings()

    PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )
    print(f"Stored {len(chunks)} chunks in Pinecone index '{PINECONE_INDEX_NAME}'.")


if __name__ == "__main__":
    main()
