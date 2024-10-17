from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# Create an instance of the OpenAI language model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.6
)

# Define the prompt template
template = """
Translate the following Thai text to English:

Thai: {thai_text}

English translation:
"""

prompt = PromptTemplate(
    input_variables=["thai_text"],
    template=template
)

# Create the translation chain
translation_chain = LLMChain(llm=llm, prompt=prompt)

# Function to translate Thai to English
def translate_thai_to_english_agent(thai_text):
    return translation_chain.run(thai_text)