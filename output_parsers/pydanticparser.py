from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct'
    ,task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class person(BaseModel):
    name: str = Field(description="The person's name")
    age : int = Field(description="The person's age")
    city : str = Field(description="The person's city")

parser = PydanticOutputParser(pydantic_object=person)

temp = PromptTemplate(
    template='generate name,age city of fictional {place} person \n. {formatinstruction}',
    input_variables=['place'],
    partial_variables={'formatinstruction': parser.get_format_instructions()}

)

prompt = temp.invoke({'place':'indian'})

result = model.invoke(prompt)
res = parser.parse(result.content)
print(res)