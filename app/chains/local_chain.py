from langchain_community.llms import HuggingFacePipeline
from transformers import  pipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

from app.models.open_thai_gpt import open_thai_gpt_model, tokenizer
from app.templates.math import question_template


pipe = pipeline(
    "text2text-generation",
    model=open_thai_gpt_model, 
    tokenizer=tokenizer, 
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# local_llm = HuggingFacePipeline(pipeline=pipe)

local_llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )