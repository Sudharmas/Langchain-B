from langchain_community.document_loaders import WebBaseLoader

url = 'https://blogs.oracle.com/java/the-arrival-of-java-25'
loader = WebBaseLoader(url)

docs = loader.load()
print(len(docs))