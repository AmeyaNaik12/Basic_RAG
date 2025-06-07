import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

//This function helps us to create embedding functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)


# Initialize the Chroma client with persistent storage at the specified path
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")

# Define the name of the collection to be used or created in the Chroma database
collection_name = "document_qa_collection"

# Retrieve the collection with the given name if it exists, otherwise create it
# Also specify the embedding function to be used for this collection
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)
