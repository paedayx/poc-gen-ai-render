from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
load_dotenv(find_dotenv())

from fastapi import FastAPI
import uvicorn

from app.handlers.chatbot import conversation_history, conversation_history_v2, getChatHistory, conversation_history_v3
from app.skl_speech_recognition.google_speech_recognition import convert_m3u8_to_wav, excecute_speech_recognition
from app.utils.video_utils import get_video_token, get_video_url
from app.vector_stores.mongodb import add_vector

from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from fastapi import Request, HTTPException
import os
from fastapi.middleware.cors import CORSMiddleware


CHANNEL_ACCESS_TOKEN = os.getenv('CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('CHANNEL_SECRET')

VECTOR_DB_NAME = "vectorDB"
CHAT_HOSTORY_DB_NAME = "vegapunk"
CHAT_HISTORY_COLLECTION = "chat_history"

app = FastAPI()

origins = os.getenv('ALLOWED_ORIGINS', '').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/learning/{course_id}/chapter/{chapter_id}/version/{video_version}/prepare-chatbot")
def prepareChatbot(course_id: int, chapter_id: int, video_version: int):
    video_token = get_video_token(course_id, chapter_id)
    m3u8_url = get_video_url(course_id, chapter_id, video_token, video_version)
    
    output_wav_path = f"app/skl_speech_recognition/wav/{course_id}-{chapter_id}.wav"
    temp_wav_path = f"app/skl_speech_recognition/wav/{course_id}-{chapter_id}-temp.wav"
    convert_m3u8_to_wav(m3u8_url, output_wav_path)
    text_result = excecute_speech_recognition(output_wav_path, temp_wav_path)

    os.remove(output_wav_path)

    add_vector(VECTOR_DB_NAME, f"course-{course_id}-chapter-{chapter_id}", f"course-{course_id}-chapter-{chapter_id}", text_result)

    return {"result": text_result}

class ChatBody(BaseModel):
    query: str
    user_id: int
    user_email: str
    course_name: str
    chapter_name: str

@app.post("/learning/{course_id}/chapter/{chapter_id}/chat")
def chat_with_bot(course_id: int, chapter_id: int, payload: ChatBody):
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

@app.post("/v2/learning/{course_id}/chapter/{chapter_id}/chat")
def chat_with_bot_v2(course_id: int, chapter_id: int, payload: ChatBody):
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
        return "เกิดข้อผิดพลาด"
    
@app.get("/conversation-history")
def getConversationHistory():
    result = getChatHistory(CHAT_HOSTORY_DB_NAME, CHAT_HISTORY_COLLECTION)
    return {"data": result}

@app.post("/chat")
def chat(query: str, user_id: int):
    return conversation_history(query=query, user_id=user_id)

@app.post("/chat_v2")
def chat(query: str, user_id: int):
    return conversation_history_v3()

# inital linebot API
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get("X-Line-Signature")

    # Get the request body as text
    body = await request.body()

    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    return JSONResponse(content={"success": True})

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text

    response = conversation_history(query=text, user_id=1)
    reply_text = response
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

if __name__ == "__main__":
    port = os.getenv('PORT', 8000)
    uvicorn.run(app, host="0.0.0.0", port=int(port))