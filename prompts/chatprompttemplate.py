from langchain_core.prompts import ChatPromptTemplate

# instead of using classes of messages we can use thi chatprompt template to create templates and place values
chattemplate = ChatPromptTemplate([
    ('system','you are a helpful {domain} expert'),
    ('human','explain in simple terms , the {topic} ')
])

prompts = chattemplate.invoke({'domain':'cricket','topic':'batting'})
print(prompts)