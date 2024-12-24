from fastapi import FastAPI

from src.core.rag_query_engine import RAGQueryEngine
from src.schemas.chat import Chat

app = FastAPI()

# Initialize the RAG engine
rag_engine = RAGQueryEngine(
    collection_name="FAQ",  # Replace with your collection name
    embedding_model_name="all-MiniLM-L6-v2",
    openai_model="gpt-3.5-turbo",  # You can change this to "gpt-4" if you have access
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat")
def chat_with_llm(body: Chat):
    user_query = body.query
    llm_response = rag_engine.query(user_query)["answer"]

    return {"bot": llm_response}
