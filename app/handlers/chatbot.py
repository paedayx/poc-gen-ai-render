from langchain.schema import HumanMessage, AIMessage
from app.chains.conversation_chain_qa import conversational_qa_chain, get_conversational_qa_chain
from app.vector_stores.mongodb import perform_similarity_search, find_all_vectors, add_documents, find_many_by, find_all_chat_histories
from langchain.docstore.document import Document
from datetime import datetime
from langchain.chains.summarize import load_summarize_chain
from app.models.open_ai import open_ai_model
from app.chains.conversation_session_chain import get_conversational_rag_chain

chat_histories = {}

CHAT_DB = 'vegapunk'
CHAT_COLLECTION = 'chat_history'
CHAT_COLLECTION_V2 = 'chat_history_v2'

def conversation_history(query: str, user_id: int):
    chat_histories = []
    if f"{user_id}" in chat_histories :
        chat_histories = chat_histories[f"{user_id}"]

    result = conversational_qa_chain.invoke(
        {
            "question": query,
            "chat_history": chat_histories
        }
    )

    chat_histories.append(HumanMessage(content=query))
    chat_histories.append(AIMessage(content=result.content))
    chat_histories[f"{user_id}"] = chat_histories

    return result.content

def transform_chat_message(chat):
    if chat["type"] == "ai":
        return AIMessage(content=chat["content"])
    else:
        return HumanMessage(content=chat["content"])

def conversation_history_v2(
        db_name: str, 
        collection_name: str, 
        index_name: str, 
        query: str, 
        user_id: int,
        user_email: str, 
        course_id: int, 
        course_name: str, 
        chapter_id: int, 
        chapter_name: str
    ):
    chat_histories = []
    search_result = perform_similarity_search(db_name, collection_name, index_name, query)

    mongo_chat_histories = find_many_by(CHAT_DB, CHAT_COLLECTION, {"user_id": user_id, "chapter_id": chapter_id, "type": "humen"}).sort("created_at", -1).limit(4)

    chat_histories_list = list(mongo_chat_histories)
    chat_histories_list.reverse()

    buff = None
    for chat in chat_histories_list:
        transformed_chat = transform_chat_message(chat)

        if buff:
            chat_histories.append(buff)
            buff = None
            continue

        if len(chat_histories) != 0 and type(chat_histories[-1]) == type(transformed_chat):
            buff = transformed_chat
            continue

        chat_histories.append(transformed_chat)

    if len(search_result) != 0:
        context = search_result
    else:
        data = find_all_vectors(db_name, collection_name)
        context = [Document(page_content=data)]

    custom_conversational_qa_chain = get_conversational_qa_chain(context)

    result = custom_conversational_qa_chain.invoke(
        {
            "question": query,
            "chat_history": chat_histories
        }
    )

    current_datetime = datetime.now()

    mongo_chat_histories = [
        {
            "user_id": user_id,
            "user_email": user_email,
            "content": query,
            "type": "humen",
            "course_id": course_id,
            "course_name": course_name,
            "chapter_id": chapter_id,
            "chapter_name": chapter_name,
            "created_at": current_datetime,
            "updated_at": current_datetime
        },
        {
            "user_id": user_id,
            "user_email": user_email,
            "content": result.content,
            "type": "ai",
            "course_id": course_id,
            "course_name": course_name,
            "chapter_id": chapter_id,
            "chapter_name": chapter_name,
            "created_at": current_datetime,
            "updated_at": current_datetime
        }
    ]

    add_documents(CHAT_DB, CHAT_COLLECTION, mongo_chat_histories)

    return result.content

def conversation_history_v3(
        db_name: str, 
        collection_name: str, 
        index_name: str, 
        query: str, 
        user_id: int,
        user_email: str, 
        course_id: int, 
        course_name: str, 
        chapter_id: int, 
        chapter_name: str
    ):
    user_question_datetime = datetime.now()

    search_result = perform_similarity_search(db_name, collection_name, index_name, query)
    if len(search_result) != 0:
        context = search_result
    else:
        data = find_all_vectors(db_name, collection_name)
        context = [Document(page_content=data)]
    conversational_rag_chain = get_conversational_rag_chain(context)

    result = conversational_rag_chain.invoke(
                {"input": query},
                config={
                    "configurable": {"session_id": f"{user_id}-{chapter_id}"}
                },
            )["answer"]
    
    ai_answer_datetime = datetime.now()

    mongo_chat_histories = [
        {
            "user_id": user_id,
            "user_email": user_email,
            "content": query,
            "type": "humen",
            "course_id": course_id,
            "course_name": course_name,
            "chapter_id": chapter_id,
            "chapter_name": chapter_name,
            "created_at": user_question_datetime,
            "updated_at": user_question_datetime
        },
        {
            "user_id": user_id,
            "user_email": user_email,
            "content": result,
            "type": "ai",
            "course_id": course_id,
            "course_name": course_name,
            "chapter_id": chapter_id,
            "chapter_name": chapter_name,
            "created_at": ai_answer_datetime,
            "updated_at": ai_answer_datetime
        }
    ]

    add_documents(CHAT_DB, CHAT_COLLECTION_V2, mongo_chat_histories)

    return result

def getChatHistory(db_name: str, collection_name: str):
    results = []
    for chat in find_all_chat_histories(db_name, collection_name):
        results.append({
            "user_id": chat["user_id"],
            "user_email": chat["user_email"],
            "content": chat["content"],
            "type": chat["type"],
            "course_id": chat["course_id"],
            "course_name": chat["course_name"],
            "chapter_id": chat["chapter_id"],
            "chapter_name": chat["chapter_name"],
            "created_at": chat["created_at"],
            "updated_at": chat["updated_at"]
        })
    return results
    
