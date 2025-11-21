from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

chattemplate = ChatPromptTemplate([
    ('system','you are helpful customer support agent'),
    MessagesPlaceholder(variable_name='chathistory'),
    ('human','{query}')
])

chathistory = [] # placehoder variable which will be placed inside our template.

with open('chat.txt') as f:
    chathistory.extend(f.readlines())

print(chathistory)

prompt = chattemplate.invoke({'chathistory':chathistory,'query':'where is my refund'})

print(prompt)