from langchain.chains import LLMChain
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

template = """Question: {question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])

hugginface_hub_chain = LLMChain(prompt=prompt, 
                                llm=HuggingFaceHub(
                                        repo_id="google/flan-t5-xxl", 
                                        model_kwargs={"temperature":1}
                                    )
                                )