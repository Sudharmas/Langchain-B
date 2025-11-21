from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct'
    ,task="text-generation",
)

model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me name and city of fictional person . \n {formatinstruction}',
    input_variables=[],
    partial_variables={'formatinstruction':parser.get_format_instructions()}
)

# prompt = template.format()
# result = model.invoke(prompt)
# res =parser.parse(result.content)

res = chain = template | model | parser
res = chain.invoke({})
print(res)
