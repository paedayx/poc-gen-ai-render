from langchain.chains import LLMChain
from langchain_community.llms import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
prompt = PromptTemplate(
    input_variables=["animal"],
    template="What is {animal}"
)

chain = LLMChain(llm=llm, prompt=prompt)

if __name__ == "__main__":
    print(chain.run("a dog"))