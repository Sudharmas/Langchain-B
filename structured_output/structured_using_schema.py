from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Optional, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
json_schem={
  "title": "reviews",
  "description": "about students",
  "type": "object",
    "name": "xyz",
    "age": "19"
  }
structuredmodel =model.with_structured_output(json_schem)

result = structuredmodel.invoke("""the hardware is great,but software is very poor.
there are many pre installed apps that cant be removes""")

# print(result)
print(result.name)
