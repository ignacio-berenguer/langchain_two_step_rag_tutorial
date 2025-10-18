from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from openai import embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Function to load documents from a PDF file
def load_documents(file_path: str ):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs

def get_embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings

def get_vector_store(embeddings):
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    return vector_store


# Main function
def db():
    print("Starting langchain-two-step-rag-tutorial!")

    print("Setting up models and vector store... ")
    # model = init_chat_model("openai:gpt-4.1")
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)


    print("Setup complete.")
    print("Starting indexing... ")
    print("Loading documents... ")
    # documents = load_documents("./data/el_capitan_alatriste.pdf")
    documents = load_documents("./data/Convenio_XXI_de_la_Industria_Química.pdf")
    print(f"Loaded {len(documents)} documents.")
    print(documents[0].page_content[:500])  # Print first 500 characters of the first document
    print(documents[0].metadata)  # Print metadata of the first document


    print("Dividing documents into chunks...")
    chunked_docs = chunk_documents(documents, chunk_size=500, chunk_overlap=100)
    print(f"Created {len(chunked_docs)} chunks.")
    print(chunked_docs[0].page_content)  # Print first 100 characters of the first chunk
    print(chunked_docs[0].metadata)  # Print metadata of the first chunk


    print("Adding documents to vector store...")
    vector_store.add_documents(documents=chunked_docs)
    print(f"Documents added to vector store.")

    print("Indexing complete.")



if __name__ == "__main__":
    db()
