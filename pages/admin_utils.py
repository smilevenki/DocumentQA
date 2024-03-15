from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings     #This import has been replaced by the below one.
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#from langchain.llms import OpenAI #This import has been replaced by the below one.
from langchain_openai import OpenAI
from pinecone import Pinecone as PineconeStore
#from langchain.vectorstores import Pinecone     #This import has been replaced by the below one.
from langchain_community.vectorstores import Pinecone
import os

#**********Functions to help load documents to PINECONE************

#Read PDF data
def read_pdf_data(pdf_file):
    pdf_page = PdfReader(pdf_file)
    text = ""
    for page in pdf_page.pages:
        text += page.extract_text()
    return text

#Split data into chunks
def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_text(text)
    docs_chunks =text_splitter.create_documents(docs)
    return docs_chunks

#Create embeddings instance
def create_embeddings_load_data():
    embeddings = OpenAIEmbeddings()
    #embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

#Function to push data to Pinecone
def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):

    PineconeStore(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index