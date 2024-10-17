from datetime import datetime
from app.chains.hr_slack_chain import get_hr_slack_chain
from app.vector_stores.mongodb import add_documents

VEGAPUNK_DB = 'vegapunk'
CHAT_COLLECTION = 'hr_slack_bot_chat_history'

def hr_slack_bot(user_id, query):
  user_question_datetime = datetime.now()
  hr_slack_chain = get_hr_slack_chain(user_id=user_id)
  result = hr_slack_chain.invoke(
                {
                    "question": query,
                    "chat_history": []
                },
            )["answer"]
  
  ai_answer_datetime = datetime.now()
  
  mongo_chat_histories = [
      {
          "user_id": user_id,
          "content": query,
          "type": "human",
          "created_at": user_question_datetime,
          "updated_at": user_question_datetime,
          
      },
      {
          "user_id": user_id,
          "content": result,
          "type": "ai",
          "created_at": ai_answer_datetime,
          "updated_at": ai_answer_datetime,
          
      }
  ]

  add_documents(VEGAPUNK_DB, CHAT_COLLECTION, mongo_chat_histories)
  
  return result