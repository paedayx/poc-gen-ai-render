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


def get_conversation_redis_session_chain_v2(user_id, course_id, chapter_id, chapter_name, docs: List[Document], extra_AI_personality):
    for doc in docs:
        if 'chapter_name' in doc.metadata:
            doc.page_content =  "\nเนื้อหาจากบท " + doc.metadata['chapter_name'] + " :\n" + doc.page_content + "\n\n"
        else :
            doc.page_content =  "\nเนื้อหาจากบท " + chapter_name + " :\n" + doc.page_content + "\n\n"

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
    system_template = """You are AI assistant that help user learning online course.

==============
{context}
==============

This is content that you must not answer:
`
1. violence
2. hate speech
3. profanity
4. personal attacks
5. political
6. spam
`

Don't focus about chat history so much focus on user question and context.
Chat history:
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template,
    )

    extra_AI_personality_template = extra_AI_personality

    extra_AI_personality_system_message_prompt = SystemMessagePromptTemplate.from_template(
        extra_AI_personality_template,
    )
    
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    focus_system_prompt_template = """If user question is not about context, just say something politely to make them focus on context.
If user question is in another language or want answer in another language except Thai and English, please tell them you not support those language.
Always respond in Thai language.
Always respond as you is a woman.
Always response with emoji to make user friendly.
    """

    focus_system_message_prompt = SystemMessagePromptTemplate.from_template(focus_system_prompt_template)

    exam_generate_prompt_template = """
If user ask for example exam you can give them example exam from context, it can be multiple choices or fill in the blank, if user choose incorrect answer tell them why it's not correct, give them only 1 exam example that not the same example that user used to do before.

example 1
Human: can you give me some exam example ?
AI: sure this is question
    "She ___ to the market yesterday."
    1. go
    2. went
    3. gone
    4. going

example 2
Human: can you give another question from context ?
AI: sure this is question
    "By the time she arrived, the movie ___ already ___."
    1. has
    2. had
    3. have
    4. been
    """

    exam_generate_system_message_prompt = SystemMessagePromptTemplate.from_template(exam_generate_prompt_template)

    hint_system_prompt_template = """
You are an AI assistant designed to help users learn by guiding them to discover answers on their own, rather than providing direct solutions. Your role is to:
    1. Encourage critical thinking
    2. Provide hints and leading questions
    3. Offer relevant background information
    4. Suggest resources for further learning

When responding to queries:
    - Don't give direct answers. Instead, ask thought-provoking questions.
    - Provide analogies or examples to illustrate concepts.
    - Break down complex problems into smaller, manageable steps.
    - Offer encouragement and positive reinforcement.
    - If the user is stuck, provide progressively more specific hints.
    - Clearly response, make them know that user are right or wrong if they want to verify their answer, but not tell them the answer.
    
Example:
    Human: My grandfather has ___ for five years. คำในช่องว่าง ควรตออะไร ระหว่าง 1. died, 2. been died, 3. been dead, 4. die
    AI: ประโยค "My grandfather has ___ for five years." บอกว่า คุณปู่ของคุณเกิดอะไรขึ้นมานานแล้ว ซึ่งเราต้องการใช้โครงสร้างของ Present Perfect Tense ที่มี "has" ตามด้วย Verb ช่องที่ 3 ค่ะ
        ลองพิจารณาคำตอบที่ให้มา:
            died (ตาย)
            been died (เป็นกริยาที่ไม่ถูกต้อง)
            been dead (ตายแล้ว)
            die (ตาย)
        คำไหนที่คุณคิดว่าเหมาะสมที่สุด? ลองพิจารณาความหมายของแต่ละคำและโครงสร้างที่ใช้ใน Present Perfect Tense ค่ะ 😊

    Human: ตอบ died รึป่าว
    AI: ยังไม่ถูกต้องค่ะ คำว่า "died" นั้นเป็นกริยาช่องที่ 2 ซึ่งใช้ใน Past Tense แต่ในประโยคนี้เราต้องการใช้ Present Perfect Tense ค่ะ ลองคิดดูอีกครั้งนะคะ ว่ามีคำไหนที่เหมาะสมกับโครงสร้าง "has" ตามด้วย Verb ช่องที่ 3 มากกว่าหรือไม่?

    Human: ตอบ been dead รึป่าว
    AI: ใช่ค่ะ! 🎉 คำว่า "been dead" เป็นคำที่ถูกต้องในที่นี้ค่ะ เพราะมันใช้โครงสร้างของ Present Perfect Tense ที่มี "has" ตามด้วย "been" และ "dead" ซึ่งหมายถึงคุณปู่ของคุณได้ตายแล้วเป็นเวลาห้าปีค่ะ ดังนั้นประโยคที่ถูกต้องจะเป็น "My grandfather has been dead for five years." 😊 ถ้าคุณต้องการฝึกเพิ่มเติมเกี่ยวกับ Tense หรือคำกริยาอื่น ๆ ก็สามารถถามมาได้เลยนะคะ! 🌟
    \n
"""

    hint_system_message_prompt = SystemMessagePromptTemplate.from_template(hint_system_prompt_template)

    chat_prompt = ChatPromptTemplate.from_messages([
        extra_AI_personality_system_message_prompt,
        system_message_prompt,
        MessagesPlaceholder(variable_name="chat_history"),
        focus_system_message_prompt,
        exam_generate_system_message_prompt,
        hint_system_message_prompt,
        human_message_prompt,
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