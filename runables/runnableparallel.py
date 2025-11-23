from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='generate a linkedin post on {text} .',
    input_variables=['text']
)

prompt1 = PromptTemplate(
    template='generate tweet on topic \n {text}',
    input_variables=['text']
)

llm1 = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct'
    , task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct'
    , task="text-generation"
)

model = ChatHuggingFace(llm=llm1)
model1 = ChatHuggingFace(llm=llm2)

parser = StrOutputParser()

parallelchain = RunnableParallel({
    'linkedin':RunnableSequence(prompt, model, parser),
    'tweet':RunnableSequence(prompt1, model1, parser),
})


result = parallelchain.invoke({'text': 'AI'})
print(result)
parallelchain.get_graph().print_ascii()