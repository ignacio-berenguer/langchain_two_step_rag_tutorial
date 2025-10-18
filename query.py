from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from openai import embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Function to query the vector store
def query_vector_store(vector_store, query: str, model, k=3):
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=k)
    return docs

def run_query():
    print("Setting up models and vector store for querying... ")
    model = init_chat_model("openai:gpt-4.1")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

    print("Setup complete.")
    query = "¿Quienes tienden una emboscada a Alatriste?"
    print(f"Running query: {query}")
    answer = query_vector_store(vector_store, query, model, k=10)
    for doc in answer:
        print(f"Answer: {doc.page_content}")


if __name__ == "__main__":
    run_query()