from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct'
    ,task="text-generation",
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

template1 = PromptTemplate(
    template='write a 5 line summary on {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template | model | parser | template1 | model | parser
result = chain.invoke({'topic':'black hole'})

print(result)

# prompt = template.invoke({'topic':'black hole'})
#
# result = model.invoke(prompt)
#
# prompt2 = template1.invoke({'text':result.content})
#
# result1 = model.invoke(prompt2)
#
# print(result1.content)
