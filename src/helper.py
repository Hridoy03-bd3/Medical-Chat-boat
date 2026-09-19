from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_file(data_path: str):
    """Load every PDF document found in the given directory."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF directory not found: {path}")

    loader = DirectoryLoader(
        str(path),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDF files found in: {path}")
    return documents


def text_split(documents, chunk_size: int = 500, chunk_overlap: int = 20):
    """Split loaded documents into retrieval-sized chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def download_hugging_face_embeddings():
    """Return the embedding model used by both indexing and retrieval."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
