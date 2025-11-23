from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel

load_dotenv()

passthrough = RunnablePassthrough()

# print(passthrough.invoke({'name':'xyz'}))


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
jokegen = RunnableSequence(prompt,model,parser)

parallelchain = RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'explain':RunnableSequence(prompt1,model,parser),
    }
)

finalchain = RunnableSequence(jokegen,parallelchain)
print(finalchain.invoke({'topic':'cricket'}))
finalchain.get_graph().print_ascii()
