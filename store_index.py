from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as PineconeStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extract_data = load_pdf('dataset/')
text_chunks = text_split(extract_data)
embeddings = download_hugging_face_embeddings()

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "vni-medical"
# Creating Embeddings for each of the text chunks and storing
docsearch = PineconeStore.from_texts(
    [t.page_content for t in text_chunks], embeddings, index_name=index_name)
