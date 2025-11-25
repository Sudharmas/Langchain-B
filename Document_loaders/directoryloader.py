from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader

loader = DirectoryLoader(
    path='documents'
    ,glob='*.pdf',
    loader_cls=PDFPlumberLoader
)

docs = loader.load()
# docs = loader.lazy_load()
print(len(docs))
print(docs[0].page_content)