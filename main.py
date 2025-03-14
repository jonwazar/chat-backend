from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # Hugging Face for embeddings
from llama_index.llms.openai import OpenAI  # OpenAI for chat responses

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict this later)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Initialize ChromaDB client and collection
db = chromadb.PersistentClient(path="./chroma_db")  # Ensure this matches Colab path
chroma_collection = db.get_or_create_collection("healthGPT")

# Assign Chroma as the vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load index from storage with Hugging Face embeddings
try:
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Update model as needed
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
except Exception as e:
    raise RuntimeError("Failed to load index from storage") from e

# Initialize OpenAI LLM for response generation
llm = OpenAI(model="gpt-3.5-turbo", api_key="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")

# Define request model
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        query = request.message
        query_engine = index.as_query_engine(llm=llm)  # Pass OpenAI LLM to query engine
        response = query_engine.query(query)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally with: uvicorn main:app --host 0.0.0.0 --port 8000
