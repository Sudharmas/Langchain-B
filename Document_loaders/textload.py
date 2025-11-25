from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct"
    ,task="text-generation",
)
model = ChatHuggingFace(llm=llm)


prompt = PromptTemplate(
    template='write a summary on {topic}',
    input_variables=['topic'],
)

parser = StrOutputParser()

loader = TextLoader('text.txt')
docs = loader.load()


chain = prompt | model | parser

print(chain.invoke({'topic': docs[0].page_content}))


