from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms import OpenAI
from llama_index.embeddings import HuggingFaceEmbedding

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Allows requests from your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://jonwazar.github.io"],  # Change "*" to restrict access to your frontend only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class ChatRequest(BaseModel):
    message: str

# Get OpenAI API Key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OpenAI API Key. Set it in Render's environment variables.")

# Initialize OpenAI LLM
llm = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Hugging Face Embedding model
hf_embeddings = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB client and collection
db = chromadb.PersistentClient(path="./chroma_db")  
chroma_collection = db.get_or_create_collection("healthGPT")

# Assign Chroma as the vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load index from storage
try:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=hf_embeddings)
except Exception as e:
    raise RuntimeError("Failed to load index from storage") from e

# Define the /chat endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        query = request.message
        query_engine = index.as_query_engine(llm=llm)  # Pass OpenAI LLM to query engine
        response = query_engine.query(query)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run Uvicorn if script is executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's PORT variable
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
