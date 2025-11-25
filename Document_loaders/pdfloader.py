from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('Cloud computing 1-8.pdf')


docs = loader.load()
print(len(docs))
# same operations afterwards...