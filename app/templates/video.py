answer_template = """You are student assistant, answer the question based on only this context:
{context}

context is transcript of video.

Question: {question}

Answer in the following language: Thai
"""

question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

don't answer same as answer before.
summary answer in 60 words.

Chat History:
"{chat_history}"
Follow Up Input: {question}
Standalone question:
"""