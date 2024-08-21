from typing import List
from dotenv import load_dotenv
load_dotenv()
import os

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6
)

URL = os.getenv('UPSTASH_REDIS_URL')
TOKEN = os.getenv('UPSTASH_REDIS_TOKEN')

def get_conversation_redis_session_chain(user_id, course_id, chapter_id, docs: List[Document]):
    # Create a vector store from the documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Set up the chat history and memory
    history = UpstashRedisChatMessageHistory(
        url=URL, token=TOKEN, ttl=500, session_id=f"LP-{user_id}-{course_id}-{chapter_id}"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=history,
    )

    # Create a custom prompt template with a system message
    system_template = """
    You are AI assistant that help user learn course chapter.
    This is course chapter transcript that user learning:
    `
    {context}
    `

    This is content that you must not answer:
    `
    1. violence
    2. hate speech
    3. profanity
    4. personal attacks
    5. political
    6. spam
    `

    please teach them, let them think step by step. don't tell them the answer.
    if user question is not about context, just say something politely to make them focus on context.
    if user question is in another language or want answer in another language except Thai and English, please tell them you not support those language.
    Always respond in Thai language.
    Always respond as you is a woman.
    Always response with emoji to make user friendly.
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        MessagesPlaceholder(variable_name="chat_history"),
        human_message_prompt
    ])

    # Create the conversational retrieval chain with the custom prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
        verbose=True,
        get_chat_history=lambda h : h,
    )
    
    return chain