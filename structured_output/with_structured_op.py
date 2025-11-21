from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
class review(TypedDict):
    key_themes: Annotated[list[str],"write down all key themes discussed in review in a list"]
    summary:Annotated[str,"A brief summary of review"]
    sentiment:Annotated[Literal["pos","neg"],"return sentiment of review either negative or poitive or neutral"]
    pros: Annotated[Optional[list[str]],"write down all pros inside lost"]
    cons: Annotated[Optional[list[str]],"write down all cons inside lost"]
    cons: Annotated[Optional[str],"write down name of reviewer"]

structuredmodel =model.with_structured_output(review)

result = structuredmodel.invoke("""the hardware is great,but software is very poor.
there are many pre installed apps that cant be removes""")

# print(result)
print(result['key_themes'])
print(result['summary'])
print(result['sentiment'])