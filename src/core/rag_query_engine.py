import os
import textwrap
from typing import List, Dict

import torch
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from src.core.prompts import generic_prompt


class RAGQueryEngine:
    def __init__(
        self,
        collection_name: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        top_k: int = 3,
        openai_model: str = "gpt-3.5-turbo",
    ):
        # Initialize Qdrant client
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.top_k = top_k
        self.openai_model = openai_model

        # Initialize embedding model for local embeddings
        self.device = torch.device("cpu")
        self.embedding_model = SentenceTransformer(
            embedding_model_name, device=self.device
        )

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def get_relevant_documents(self, query: str) -> List[Dict]:
        """Retrieve relevant documents from Qdrant."""
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=self.top_k,
        )

        # Extract documents and their scores
        documents = []
        for result in search_result:
            documents.append(
                {
                    "content": result.payload["content"],
                    "metadata": result.payload["metadata"],
                    "score": result.score,
                }
            )

        return documents

    def format_messages(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Format the conversation messages with retrieved documents."""
        context = "\n\n".join(
            [
                f"Document {i + 1} (Source: {doc['metadata']['source']}):\n{doc['content']}"
                for i, doc in enumerate(documents)
            ]
        )

        messages = [
            {"role": "system", "content": generic_prompt},
            {
                "role": "user",
                "content": f"""Please answer the following question based on the provided context:

Context:
{context}

Question: {query}""",
            },
        ]

        return messages

    def generate_answer(self, messages: List[Dict]) -> str:
        """Generate answer using OpenAI's API."""
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=0.3,
            max_tokens=500,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
        )

        return response.choices[0].message.content

    def query(self, query: str, verbose: bool = False) -> Dict:
        """Main method to process a query and return an answer."""
        # Get relevant documents
        documents = self.get_relevant_documents(query)

        # Format messages
        messages = self.format_messages(query, documents)

        # Generate answer
        answer = self.generate_answer(messages)

        result = {"answer": answer, "source_documents": documents}

        if verbose:
            print("\nRetrieved Documents:")
            for i, doc in enumerate(documents):
                print(f"\nDocument {i + 1}")
                print(f"Source: {doc['metadata']['source']}")
                print(f"Relevance Score: {doc['score']}")
                print("Content:", textwrap.fill(doc["content"], width=80))

            print("\nGenerated Answer:")
            print(textwrap.fill(answer, width=80))

        return result


def main():
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Initialize the RAG engine
    rag_engine = RAGQueryEngine(
        collection_name="FAQ",  # Replace with your collection name
        embedding_model_name="all-MiniLM-L6-v2",
        openai_model="gpt-3.5-turbo",  # You can change this to "gpt-4" if you have access
    )

    # Interactive query loop
    print("RAG Query System (type 'exit' to quit)")
    print("-" * 50)

    while True:
        query = input("\nEnter your question: ")
        if query.lower() == "exit":
            break

        try:
            print("\nProcessing query...")
            result = rag_engine.query(query, verbose=True)
        except Exception as e:
            print(f"Error processing query: {str(e)}")


if __name__ == "__main__":
    main()
