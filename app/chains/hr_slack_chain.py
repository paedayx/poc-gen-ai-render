from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
load_dotenv()
import os

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from app.documents.skilllane_hr_doc import hr_document
from langchain.docstore.document import Document

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.6
)

URL = os.getenv('UPSTASH_REDIS_URL')
TOKEN = os.getenv('UPSTASH_REDIS_TOKEN')

def get_hr_slack_chain(user_id):
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    
    docs = text_splitter.split_documents([Document(page_content=hr_document)])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Set up the chat history and memory
    history = UpstashRedisChatMessageHistory(
        url=URL, token=TOKEN, ttl=500, session_id=f"HR-SLACK-BOT-{user_id}"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=history,
    )

    # Create a custom prompt template with a system message
    system_template = """You are assistant that answer about SkillLane company data like HR.

this is company data:
==============
{context}
==============

Do not try to make up an answer:
    - If the answer to the question cannot be determined from the context alone, say "I cannot determine the answer to that."
    - If the context is empty, just say "I do not know the answer to that."

This is content that you must not answer:
`
1. violence
2. hate speech
3. profanity
4. personal attacks
5. political
6. spam
`

If user question is not about context, just say something politely to make them focus on context.
If user question is in another language or want answer in another language except Thai and English, please tell them you not support those language.
Always respond in Thai language.
Always respond as you is a woman.
Always response with emoji to make user friendly.
\n
"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template,
    )

    chat_history_template = """This is chat history but don't focus on it to much.
```
{chat_history}
```
\n
"""
    chat_history_message_prompt = SystemMessagePromptTemplate.from_template(chat_history_template)


    human_template = """{question}
Assistant: 
"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        chat_history_message_prompt,
        human_message_prompt,
    ])

    # Create the conversational retrieval chain with the custom prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
    )
    
    return chain