import os
from src.helper import download_hugging_face_embeddings, filter_to_minimal_docs, load_pdf_files, text_split
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

if os.path.exists("./data/extracted_data.pkl"):
    import pickle
    with open("./data/extracted_data.pkl", "rb") as f:
        extracted_data = pickle.load(f)
else:
    extracted_data = load_pdf_files("./data")

minimal_docs = filter_to_minimal_docs(extracted_data)

text_chunks = text_split(minimal_docs)
print(f"Number of text chunks created: {len(text_chunks)}")

embedding = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=index_name
)
    