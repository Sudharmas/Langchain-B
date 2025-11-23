from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
load_dotenv()

prompt = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic'],
)
prompt1 = PromptTemplate(
    template='explain this joke about {topic}',
    input_variables=['topic'],
)
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct"
    ,task="text-generation",
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = RunnableSequence(prompt,model,parser,prompt1,model,parser)
print(chain.invoke({'topic':'AI'}))