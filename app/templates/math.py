answer_template = """Answer the question based your math knowledge and on the following context:
{context}

Question: {question}

Answer in the following language: Thai
"""

question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""