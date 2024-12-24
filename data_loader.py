import hashlib
import os
from pathlib import Path
from typing import List, Dict

import torch
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Force CPU usage
        self.device = torch.device("cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Add show_progress_bar for longer batches
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])[0]
        return embedding.tolist()


def get_document_loader(file_path: str):
    """Return appropriate loader based on file extension."""
    extension = file_path.split(".")[-1].lower()

    if extension == "pdf":
        return PyPDFLoader(file_path)
    elif extension == "txt":
        return TextLoader(file_path)
    elif extension in ["doc", "docx"]:
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def process_documents(file_paths: List[str]) -> List[Dict]:
    """Process documents and return chunks with metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    documents = []
    print("Processing documents...")
    for file_path in tqdm(file_paths, desc="Loading files"):
        try:
            loader = get_document_loader(file_path)
            doc = loader.load()
            chunks = text_splitter.split_documents(doc)

            # Add metadata
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "source": file_path,
                        "chunk_hash": hashlib.md5(
                            chunk.page_content.encode()
                        ).hexdigest(),
                    }
                )
                documents.append(
                    {"content": chunk.page_content, "metadata": chunk.metadata}
                )

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    return documents


def create_collection_if_not_exists(client: QdrantClient, collection_name: str):
    """Create a new collection if it doesn't exist."""
    collections = client.get_collections().collections
    if not any(collection.name == collection_name for collection in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=384,  # all-MiniLM-L6-v2 embedding dimension
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Created new collection: {collection_name}")


def main():
    # Print CPU information
    print("Using device: CPU")
    print(f"Number of available CPU threads: {os.cpu_count()}")

    # Initialize Qdrant client and embeddings model
    client = QdrantClient("localhost", port=6333)
    embeddings = SentenceTransformerEmbeddings()
    KNOWLEDGE_BASE_PATH = "./data"
    # Walk through the directory
    base_path = Path(KNOWLEDGE_BASE_PATH)

    # Get all folders first
    folders = [f for f in base_path.iterdir() if f.is_dir()]

    for folder in tqdm(folders, desc="Processing folders"):
        collection_name = folder.name
        print(f"\nProcessing folder: {collection_name}")

        # Get all supported files in the folder
        supported_extensions = {".pdf", ".txt", ".doc", ".docx"}
        files = [
            str(file)
            for file in folder.rglob("*")
            if file.suffix.lower() in supported_extensions
        ]

        if not files:
            print(f"No supported documents found in {collection_name}")
            continue

        # Create collection if it doesn't exist
        create_collection_if_not_exists(client, collection_name)

        # Process documents
        documents = process_documents(files)

        # Generate embeddings and upload to Qdrant
        total_batches = (
                                len(documents) + 99
                        ) // 100  # Calculate total number of batches
        for i in tqdm(
                range(0, len(documents), 100), desc="Uploading batches", total=total_batches
        ):
            batch = documents[i: i + 100]
            texts = [doc["content"] for doc in batch]
            metadata = [doc["metadata"] for doc in batch]

            # Generate embeddings
            print(f"\nGenerating embeddings for batch {i // 100 + 1}/{total_batches}")
            embeddings_batch = embeddings.embed_documents(texts)

            # Upload to Qdrant
            client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=[hashlib.md5(text.encode()).hexdigest() for text in texts],
                    vectors=embeddings_batch,
                    payloads=[
                        {"content": text, "metadata": meta}
                        for text, meta in zip(texts, metadata)
                    ],
                ),
            )

            print(
                f"Successfully uploaded batch {i // 100 + 1}/{total_batches} to {collection_name}"
            )


if __name__ == "__main__":
    main()
