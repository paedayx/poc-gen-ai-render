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


def get_conversation_redis_session_chain_v2(user_id, course_id, chapter_id, chapter_name, docs: List[Document], extra_AI_personality, platform: str):
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
        url=URL, token=TOKEN, ttl=500, session_id=f"{platform}-{user_id}-{course_id}-{chapter_id}"
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

    extra_AI_personality_template = extra_AI_personality

    extra_AI_personality_system_message_prompt = SystemMessagePromptTemplate.from_template(
        extra_AI_personality_template,
    )

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
    
\n
    """

    exam_generate_system_message_prompt = SystemMessagePromptTemplate.from_template(exam_generate_prompt_template)

    chat_history_template = """This is chat history but don't focus on it to much.
```
{chat_history}
```
\n
"""
    chat_history_message_prompt = SystemMessagePromptTemplate.from_template(chat_history_template)

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
    
Example 1:
    Human: The children ___ in the park all afternoon. คำในช่องว่าง ควรใส่อะไร ระหว่าง 1. play, 2. have played, 3. have been playing, 4. played
    AI: ประโยค "The children ___ in the park all afternoon." บอกว่าเด็กๆ ทำอะไรในสวนสาธารณะตลอดบ่าย เราต้องพิจารณาว่าควรใช้ Tense ไหนให้เหมาะสมกับสถานการณ์นี้ค่ะ
        มาดูตัวเลือกกันนะคะ:
            play (เล่น - Present Simple)
            have played (ได้เล่นแล้ว - Present Perfect)
            have been playing (กำลังเล่นมาตลอด - Present Perfect Continuous)
            played (เล่น - Past Simple)
        คุณคิดว่าตัวเลือกไหนเหมาะสมที่สุดสำหรับสถานการณ์นี้? ลองพิจารณาความหมายและการใช้งานของแต่ละ Tense ค่ะ 😊
    Human: ตอบ have played ถูกมั้ย
    AI: ยังไม่ถูกต้องทีเดียวค่ะ "have played" เป็น Present Perfect Tense จริง แต่ในกรณีนี้เรามีคำบอกระยะเวลา "all afternoon" ซึ่งบ่งบอกถึงการกระทำที่ต่อเนื่องมาจนถึงปัจจุบัน ลองพิจารณาดูอีกครั้งว่ามี Tense ไหนที่เหมาะสมกว่าสำหรับการแสดงถึงการกระทำที่ต่อเนื่องมาตลอดช่วงเวลาหนึ่งจนถึงปัจจุบันค่ะ
    Human: ขอเฉลยหน่อยได้มั้ย
    AI: ฉันไม่สามารถให้คำตอบได้ค่ะ
    Human: งั้นตอบ have been playing ดีกว่า
    AI: ถูกต้องแล้วค่ะ! 🎉 "have been playing" เป็นคำตอบที่เหมาะสมที่สุดในสถานการณ์นี้ค่ะ เพราะมันเป็น Present Perfect Continuous Tense ซึ่งใช้เพื่อแสดงการกระทำที่เริ่มในอดีตและดำเนินต่อเนื่องมาจนถึงปัจจุบัน ประโยคที่สมบูรณ์จะเป็น "The children have been playing in the park all afternoon." หมายความว่าเด็กๆ ได้เล่นอยู่ในสวนมาตลอดบ่ายจนถึงตอนนี้ค่ะ Present Perfect Continuous Tense เหมาะสำหรับสถานการณ์ที่มีการบอกระยะเวลาแบบต่อเนื่อง เช่น "all afternoon", "for hours", "since morning" เป็นต้น  ถ้าคุณอยากฝึกเพิ่มเติมเกี่ยวกับ Tense ต่างๆ หรือมีคำถามอื่นๆ ก็ถามมาได้เลยนะคะ! 😊

Example 2:
    Human: If I ___ harder, I would have passed the exam. คำในช่องว่าง ควรใส่อะไร ระหว่าง 1. study, 2. studied, 3. had studied, 4. have studied
    AI: ประโยค "If I ___ harder, I would have passed the exam." เป็นประโยคเงื่อนไขที่แสดงถึงสถานการณ์ในอดีตที่ไม่ได้เกิดขึ้นจริง เราเรียกว่า Third Conditional ค่ะ
        มาดูตัวเลือกกันนะคะ:
            study (เรียน - Present Simple)
            studied (เรียน - Past Simple)
            had studied (ได้เรียนแล้ว - Past Perfect)
            have studied (ได้เรียนแล้ว - Present Perfect)
        คุณคิดว่าตัวเลือกไหนเหมาะสมที่สุดสำหรับโครงสร้างประโยค Third Conditional? ลองพิจารณาความหมายและการใช้งานของแต่ละ Tense ค่ะ 🤔
    Human: ตอบ studied ได้ไหม
    AI: ใกล้เคียงแล้วค่ะ แต่ยังไม่ใช่คำตอบที่ถูกต้องที่สุด "studied" เป็น Past Simple Tense จริง แต่ในโครงสร้าง Third Conditional เราต้องการ Tense ที่แสดงถึงเหตุการณ์ที่เกิดก่อนอีกเหตุการณ์หนึ่งในอดีต ลองคิดดูอีกครั้งนะคะว่ามี Tense ไหนที่แสดงถึงการกระทำที่เกิดขึ้นก่อนอีกเหตุการณ์หนึ่งในอดีต? 🕰️
    Human: ขอเฉลยหน่อยได้มั้ย
    AI: ฉันไม่สามารถให้คำตอบได้ค่ะ
    Human: งั้นตอบ had studied ดีกว่า
    AI: ถูกต้องแล้วค่ะ! 🎉👏 "had studied" เป็นคำตอบที่ถูกต้องที่สุดในกรณีนี้ ในโครงสร้าง Third Conditional เราใช้ Past Perfect (had + past participle) ในส่วนของ if-clause เพื่อแสดงถึงเหตุการณ์ที่ไม่ได้เกิดขึ้นจริงในอดีต ดังนั้น ประโยคที่สมบูรณ์จะเป็น: "If I had studied harder, I would have passed the exam." ความหมายคือ "ถ้าฉันได้ตั้งใจเรียนมากกว่านี้ (แต่ความจริงไม่ได้ทำ) ฉันก็คงจะสอบผ่านไปแล้ว" Third Conditional ใช้เพื่อพูดถึงสถานการณ์ในอดีตที่เป็นไปไม่ได้แล้ว และผลลัพธ์ที่อาจเกิดขึ้นถ้าสถานการณ์นั้นเป็นจริง เก่งมากค่ะที่ตอบได้ถูกต้อง! 😊 ถ้าคุณมีคำถามเพิ่มเติมเกี่ยวกับ Conditional Sentences หรือหัวข้ออื่นๆ ก็ถามมาได้เลยนะคะ
==============================================================================================================
\n
"""

    hint_system_message_prompt = SystemMessagePromptTemplate.from_template(hint_system_prompt_template)

    human_template = """{question}
Assistant: 
"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        exam_generate_system_message_prompt,
        chat_history_message_prompt,
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
    )
    
    return chain