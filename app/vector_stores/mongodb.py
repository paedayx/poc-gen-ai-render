import os
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain.docstore.document import Document
import logging

import certifi
ca = certifi.where()

ATLAS_CONNECTION_STRING = os.getenv('ATLAS_CONNECTION_STRING')
INDEX_NAME = "default"
cluster = MongoClient(ATLAS_CONNECTION_STRING)
try:
    cluster.admin.command('ping')
    logging.info("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    logging.error(e)

def add_vector(db_name: str, collection_name: str, index_name: str, text: str):
    doc = Document(page_content=text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents([doc])

    collection = cluster[db_name][collection_name]

    # TODO: using collection.create_search_index() by index_name variable before created vector search

    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        collection=collection,
        index_name=index_name
    )

    return vector_search

def create_vector_search(db_name: str, collection_name: str, index_name: str):
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        ATLAS_CONNECTION_STRING,
        f"{db_name}.{collection_name}",
        OpenAIEmbeddings(),
        index_name=index_name
    )
    return vector_search

def perform_similarity_search(db_name: str, collection_name: str, index_name: str, query, top_k=3):
    vector_search = create_vector_search(db_name, collection_name, index_name)
    results = vector_search.similarity_search(
        query=query,
        k=top_k,
    )
    
    return results

def find_all_vectors(db_name: str, collection_name: str):
    all_data = cluster[db_name][collection_name].find()
    text_list = [document["text"] for document in all_data]
    joined_text = " ".join(text_list)
    return joined_text

def find_all_chat_histories(db_name: str, collection_name: str):
    return cluster[db_name][collection_name].find()

def find_many_by(db_name: str, collection_name: str, query: str):
    return cluster[db_name][collection_name].find(query)

def add_documents(db_name: str, collection_name: str, documents: list):
    collection = cluster[db_name][collection_name]
    collection.insert_many(documents)