from langchain.schema import format_document

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain.prompts.prompt import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# model
from app.models.open_ai import open_ai_model

from app.documents.sample_doc import skl_new_info
from app.documents.subtitle import subtitle
# from app.templates.math import question_template, answer_template
from app.templates.video import question_template, answer_template

vectorstore = FAISS.from_texts(
    [subtitle], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(question_template)


ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | open_ai_model
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}
conversational_qa_chain = _inputs | _context  | ANSWER_PROMPT  | open_ai_model

def get_conversational_qa_chain(docs):
    custom_vectorstore = FAISS.from_documents(
        docs, embedding=OpenAIEmbeddings()
    )
    custom_retriever = custom_vectorstore.as_retriever()

    _context = {
        "context": itemgetter("standalone_question") | custom_retriever | _combine_documents,
        "question": lambda x: x["standalone_question"],
    }
    custom_conversational_qa_chain = _inputs | _context  | ANSWER_PROMPT  | open_ai_model

    return custom_conversational_qa_chain