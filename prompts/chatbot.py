from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct'
    ,task="text-generation",
)

model = ChatHuggingFace(llm=llm)

chathistory =[
    SystemMessage(content="you are a helpful assistant")
]

while True:
    user = input('you: ')
    if user == 'exit':
        break
    chathistory.append(HumanMessage(content=user))
    result = model.invoke(chathistory)
    chathistory.append(AIMessage(content=result.content))
    print(result.content)

print(chathistory)

