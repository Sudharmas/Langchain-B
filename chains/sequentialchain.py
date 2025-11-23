
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt1 = PromptTemplate(
    template='give me a five point summary on text > \n {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct'
    , task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt1 | model | parser

result = chain.invoke({'topic':'unemployment','text':prompt.invoke('topic')})

chain.get_graph().print_ascii()

print(result)