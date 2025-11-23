from langchain_core.runnables import  RunnableBranch,RunnableLambda #converts lambda into runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

# model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='give the Sentiment of the feedback')

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template=(
        'Classify the sentiment of the following feedback text into positive or negative.\n {feedback} \n {format_instructions}'
    ),
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()},
)

prompt2 = PromptTemplate(
    template='write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


classifychain = prompt | model | parser2

result = classifychain.invoke({'feedback': 'this is a terrible one'}).sentiment
print(result)

branchchain = RunnableBranch(
    (lambda x:x.sentiment == 'positive',prompt2 | model | parser1),
    (lambda x:x.sentiment == 'negative',prompt3 | model | parser1),
    RunnableLambda(lambda x:"couldnot find sentiment")
)

chain = classifychain | branchchain
result = chain.invoke({'feedback': 'this is a terrible one'})
print(result)

chain.get_graph().print_ascii()