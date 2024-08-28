from langchain.schema import HumanMessage, AIMessage
from app.chains.conversation_chain_qa import get_conversational_qa_chain
from app.vector_stores.mongodb import find_one, perform_similarity_search, find_all_vectors, add_documents, find_many_by, find_all_chat_histories, update_document
from langchain.docstore.document import Document
from datetime import datetime
from app.chains.conversation_session_chain import get_conversational_rag_chain
from app.chains.conversation_redis_session_chain import get_conversation_redis_session_chain, get_conversation_redis_session_chain_v2
from app.chains.translation_chain import translate_thai_to_english

chat_histories = {}

VECTOR_DB = 'vectorDB'
VEGAPUNK_DB = 'vegapunk'
CHAT_COLLECTION = 'chat_history'
CHAT_COLLECTION_V2 = 'chat_history_v2'
CHAT_COLLECTION_V3 = 'chat_history_v3'
CSAT_COLLECTION = "csat"
EXTRA_AI_PERSONALITY_COLLECTION = "extra_AI_personality"

def transform_chat_message(chat):
    if chat["type"] == "ai":
        return AIMessage(content=chat["content"])
    else:
        return HumanMessage(content=chat["content"])

def conversation_history_v1(
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

    mongo_chat_histories = find_many_by(VEGAPUNK_DB, CHAT_COLLECTION, {"user_id": user_id, "chapter_id": chapter_id, "type": "humen"}).sort("created_at", -1).limit(4)

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

    add_documents(VEGAPUNK_DB, CHAT_COLLECTION, mongo_chat_histories)

    return result.content

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

    add_documents(VEGAPUNK_DB, CHAT_COLLECTION_V2, mongo_chat_histories)

    return result

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
    conversational_rag_chain = get_conversation_redis_session_chain(user_id=user_id, course_id=course_id, chapter_id=chapter_id, docs=context)

    result = conversational_rag_chain.invoke(
                {
                    "question": query,
                    "chat_history": []
                },
            )["answer"]
    
    ai_answer_datetime = datetime.now()

    mongo_chat_histories = [
        {
            "user_id": user_id,
            "user_email": user_email,
            "content": query,
            "type": "human",
            "course_id": course_id,
            "course_name": course_name,
            "chapter_id": chapter_id,
            "chapter_name": chapter_name,
            "created_at": user_question_datetime,
            "updated_at": user_question_datetime,
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
            "updated_at": ai_answer_datetime,

        }
    ]

    doc_id_list = add_documents(VEGAPUNK_DB, CHAT_COLLECTION_V3, mongo_chat_histories)

    return {
        "ai_response": result,
        "human_chat_id": str(doc_id_list[0]),
        "ai_chat_id": str(doc_id_list[1])
    }

def conversation_history_v4(
        query: str, 
        user_id: int,
        user_email: str, 
        course_id: int, 
        course_name: str, 
        chapter_id: int, 
        chapter_name: str
    ):
    user_question_datetime = datetime.now()

    query_trans:str = translate_thai_to_english(query)
    course_context_wording_list = ["course", "which chapter"]
    is_course_query: bool = any(word in query_trans.lower() for word in course_context_wording_list)

    if(is_course_query) :
        collection_name = f"course-{course_id}"
        index_name = f"course-{course_id}"
    else :
        collection_name = f"course-{course_id}-chapter-{chapter_id}"
        index_name = f"course-{course_id}-chapter-{chapter_id}"

    search_result = perform_similarity_search(VECTOR_DB, collection_name, index_name, query)

    if len(search_result) != 0:
        """
            If search by course context, it always return search result.
            If it not return search result that's mean that collection didn't have 'vector index' please set it.
        """
        context = search_result
        
    else:
        data = find_all_vectors(VECTOR_DB, collection_name)
        context = [Document(page_content=data)]

    extra_AI_personality = find_one(VEGAPUNK_DB, EXTRA_AI_PERSONALITY_COLLECTION, {"course_id": course_id})

    conversational_rag_chain = get_conversation_redis_session_chain_v2(
        user_id=user_id, 
        course_id=course_id, 
        chapter_id=chapter_id, 
        chapter_name=chapter_name,
        docs=context,
        extra_AI_personality= extra_AI_personality["prompt"] if extra_AI_personality else ""
    )

    result = conversational_rag_chain.invoke(
                {
                    "question": query,
                    "chat_history": []
                },
            )["answer"]
    
    ai_answer_datetime = datetime.now()

    mongo_chat_histories = [
        {
            "user_id": user_id,
            "user_email": user_email,
            "content": query,
            "type": "human",
            "course_id": course_id,
            "course_name": course_name,
            "chapter_id": chapter_id,
            "chapter_name": chapter_name,
            "created_at": user_question_datetime,
            "updated_at": user_question_datetime,
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
            "updated_at": ai_answer_datetime,
        }
    ]

    doc_id_list = add_documents(VEGAPUNK_DB, CHAT_COLLECTION_V3, mongo_chat_histories)

    return {
        "ai_response": result,
        "human_chat_id": str(doc_id_list[0]),
        "ai_chat_id": str(doc_id_list[1])
    }

def get_user_csat(user_id: int, course_id: int, chapter_id: int):
    results = []
    for doc in find_many_by(VEGAPUNK_DB, CSAT_COLLECTION, {"user_id": user_id, "course_id": course_id, "chapter_id": chapter_id}):
        results.append({
            "score": doc['score'],
            "chat_id_list": doc['chat_id_list'],
            "chat_collection_name": doc['chat_collection_name'],
            "user_id": doc['user_id'],
            "user_email": doc['user_email'],
            "course_id": doc['course_id'],
            "course_name": doc['course_name'],
            "chapter_id": doc['chapter_id'],
            "chapter_name": doc['chapter_name'],
            "created_at": doc['created_at'],
            "updated_at": doc['updated_at'],
        })
        
    return results

def set_chat_csat(
        score: int,
        chat_id_list: list[str],
        user_id: int,
        user_email: str, 
        course_id: int, 
        course_name: str, 
        chapter_id: int, 
        chapter_name: str,
        detail: str,
):
    current = datetime.now()
    if detail:
        doc_id = update_document(db_name=VEGAPUNK_DB, collection_name=CSAT_COLLECTION, filter={"chapter_id": chapter_id, "course_id": course_id, "user_id": user_id}, update_data={"detail": detail})
    else:
        mongo_chat_histories = [
            {
                "score": score,
                "chat_id_list": chat_id_list,
                "chat_collection_name": CHAT_COLLECTION_V3,
                "user_id": user_id,
                "user_email": user_email,
                "course_id": course_id,
                "course_name": course_name,
                "chapter_id": chapter_id,
                "chapter_name": chapter_name,
                "created_at": current,
                "updated_at": current,
            },
        ]
        doc_id = add_documents(VEGAPUNK_DB, CSAT_COLLECTION, mongo_chat_histories)[0]

    return str(doc_id)

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
    
