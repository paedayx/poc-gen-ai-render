from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# Create an instance of the OpenAI language model
llm = ChatOpenAI(
    model="ft:gpt-4o-mini-2024-07-18:personal:course-segmetation-v4:A139iePO",
    temperature=1
)

# Define the prompt template
template = """
You are an AI that determines if the user's question is related to the course or not. Answer only with true or false.

Question: {question}
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# Create the translation chain
course_segmentation_chain = LLMChain(llm=llm, prompt=prompt)

# Function to translate Thai to English
def is_question_related_to_course(question) -> str:
    return course_segmentation_chain.run(question)