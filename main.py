import secrets
from typing import Union
from dotenv import load_dotenv, find_dotenv
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from app.handlers.hr_slack_bot import hr_slack_bot
load_dotenv(find_dotenv())

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from app.handlers.chatbot import conversation_history_v1, getChatHistory, conversation_history_v2, conversation_history_v3, conversation_history_v4, set_chat_csat, get_user_csat
from app.skl_speech_recognition.google_speech_recognition import convert_m3u8_to_wav, execute_speech_recognition
from app.utils.video_utils import get_video_token, get_video_url
from app.vector_stores.mongodb import add_vector

from slack_bolt import App as SlackApp
from slack_bolt.adapter.socket_mode import SocketModeHandler as SlackSocketModeHandler
from slack_sdk import WebClient

import os
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading

VECTOR_DB_NAME = "vectorDB"
CHAT_HISTORY_DB_NAME = "vegapunk"
CHAT_COLLECTION_V3 = 'chat_history_v3'

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

origins = os.getenv('ALLOWED_ORIGINS', '').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("API_DOC_USERNAME")
    correct_password = os.getenv("API_DOC_PASSWORD")
    if not (secrets.compare_digest(credentials.username, correct_username) and
            secrets.compare_digest(credentials.password, correct_password)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.get("/docs", include_in_schema=False)
async def get_documentation(username: str = Depends(get_current_username)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")

@app.get("/openapi.json", include_in_schema=False)
async def openapi(username: str = Depends(get_current_username)):
    return get_openapi(title="FastAPI", version="0.1.0", routes=app.routes)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/learning/{course_id}/chapter/{chapter_id}/version/{video_version}/prepare-chatbot")
def prepareChatbot(course_id: int, chapter_id: int, video_version: int):
    items = [
]

    for item in items:
        course_id = item["course_id"]
        chapter_id = item["chapter_id"]
        chapter_name = item["chapter_name"]
        video_version = item["video_version"]
        video_token = get_video_token(course_id, chapter_id)
        m3u8_url = get_video_url(course_id, chapter_id, video_token, video_version)
        
        output_wav_path = f"app/skl_speech_recognition/wav/{course_id}-{chapter_id}.wav"
        temp_wav_path = f"app/skl_speech_recognition/wav/{course_id}-{chapter_id}-temp.wav"
        convert_m3u8_to_wav(m3u8_url, output_wav_path)
        result = execute_speech_recognition(output_wav_path, temp_wav_path)

        os.remove(output_wav_path)

        add_vector(
            db_name=VECTOR_DB_NAME, 
            collection_name=f"course-{course_id}-chapter-{chapter_id}", 
            index_name=f"course-{course_id}-chapter-{chapter_id}", 
            text=result,
            metadata={"course_id": course_id, "chapter_id": chapter_id, "chapter_name": chapter_name},
        )

    return {"result": True}

class ChatBody(BaseModel):
    query: str
    user_id: int
    user_email: str
    course_name: str
    chapter_name: str

@app.post("/v1/learning/{course_id}/chapter/{chapter_id}/chat")
def chat_with_bot(course_id: int, chapter_id: int, payload: ChatBody):
    try:
        collection_name = f"course-{course_id}-chapter-{chapter_id}"
        index_name = f"course-{course_id}-chapter-{chapter_id}"
        return conversation_history_v1(
            VECTOR_DB_NAME, 
            collection_name, 
            index_name, 
            query=payload.query, 
            user_id=payload.user_id, 
            user_email=payload.user_email,
            course_id=course_id, 
            course_name=payload.course_name, 
            chapter_id=chapter_id, 
            chapter_name=payload.chapter_name
        )
    except Exception as e:
        print(e)
        return "เกิดข้อผิดพลาด"

@app.post("/v2/learning/{course_id}/chapter/{chapter_id}/chat")
def chat_with_bot_v2(course_id: int, chapter_id: int, payload: ChatBody):
    try:
        collection_name = f"course-{course_id}-chapter-{chapter_id}"
        index_name = f"course-{course_id}-chapter-{chapter_id}"
        return conversation_history_v2(
            VECTOR_DB_NAME, 
            collection_name, 
            index_name, 
            query=payload.query, 
            user_id=payload.user_id, 
            user_email=payload.user_email,
            course_id=course_id, 
            course_name=payload.course_name, 
            chapter_id=chapter_id, 
            chapter_name=payload.chapter_name
        )
    except Exception as e:
        print(e)
        return "เกิดข้อผิดพลาด"

@app.post("/v3/learning/{course_id}/chapter/{chapter_id}/chat")
def chat_with_bot_v3(course_id: int, chapter_id: int, payload: ChatBody):
    try:
        collection_name = f"course-{course_id}-chapter-{chapter_id}"
        index_name = f"course-{course_id}-chapter-{chapter_id}"
        return conversation_history_v3(
            VECTOR_DB_NAME, 
            collection_name, 
            index_name, 
            query=payload.query, 
            user_id=payload.user_id, 
            user_email=payload.user_email,
            course_id=course_id, 
            course_name=payload.course_name, 
            chapter_id=chapter_id, 
            chapter_name=payload.chapter_name
        )
    except Exception as e:
        print(e)
        return {
            "ai_response": "เกิดข้อผิดพลาด"
        }
    
@app.post("/v4/learning/{course_id}/chapter/{chapter_id}/chat")
def chat_with_bot_v4(course_id: int, chapter_id: int, payload: ChatBody):
    try:
        return conversation_history_v4(
            query=payload.query, 
            user_id=payload.user_id, 
            user_email=payload.user_email,
            course_id=course_id, 
            course_name=payload.course_name, 
            chapter_id=chapter_id, 
            chapter_name=payload.chapter_name
        )
    except Exception as e:
        print(e)
        return {
            "ai_response": "เกิดข้อผิดพลาด"
        }

@app.get("/v1/learning/{course_id}/chapter/{chapter_id}/csat/user/{user_id}")
def check_csat_exist(course_id: int, chapter_id: int, user_id: int):
    try:
        docs = get_user_csat(
            course_id=course_id,
            chapter_id=chapter_id,
            user_id=user_id
        )
        return {
            "is_exist": len(docs) > 0
        }
    except Exception as e:
        print(e)
        return e
    
class Create_CSAT_Body(BaseModel):
    user_id: int
    user_email: str
    course_name: str
    chapter_name: str
    score: Union[int, None] = None
    chat_id_list: Union[list[str], None] = None
    detail: Union[str, None] = None

@app.post("/v1/learning/{course_id}/chapter/{chapter_id}/csat")
def csat(course_id: int, chapter_id: int, payload: Create_CSAT_Body):
    try:
        doc_id = set_chat_csat(
            score=payload.score,
            chat_id_list=payload.chat_id_list,
            user_id=payload.user_id, 
            user_email=payload.user_email,
            course_id=course_id, 
            course_name=payload.course_name, 
            chapter_id=chapter_id, 
            chapter_name=payload.chapter_name,
            detail=payload.detail,
        )
        return {
            "csat_id": doc_id
        }
    except Exception as e:
        print(e)
        return e
    
@app.get("/conversation-history")
def getConversationHistory():
    result = getChatHistory(CHAT_HISTORY_DB_NAME, CHAT_COLLECTION_V3)
    return {"data": result}

@app.post("/chat_v2")
def chat(query: str, user_id: int):
    return conversation_history_v2()

SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')

slack_app = SlackApp(token=SLACK_APP_TOKEN)
slack_client = WebClient(token=SLACK_APP_TOKEN)

#Message handler for Slack
@slack_app.message(".*")
def message_handler(message, say, logger):
    user_id = message['user']
    query = message['text']
    result = slack_client.users_info(user=user_id)

    user_email = ""
      
    if result['ok']:
        user_profile = result['user']['profile']
        user_email = user_profile.get('email', 'No email found')
    else:
        print(f"Could not fetch user profile for user {user_id}")
    output = hr_slack_bot(user_id=user_id, query=query, user_email=user_email)
    say(output)

if __name__ == "__main__":
    print("this is __main__")
    # Create a thread for the Slack app
    slack_thread = threading.Thread(
        target=lambda: SlackSocketModeHandler(slack_app, SLACK_BOT_TOKEN).start(),
        daemon=True  # This ensures the thread will shut down with the main program
    )
    
    # Start the Slack thread
    slack_thread.start()
    
    port = os.getenv('PORT', 8000)
    uvicorn.run(app, host="0.0.0.0", port=int(port))
    
