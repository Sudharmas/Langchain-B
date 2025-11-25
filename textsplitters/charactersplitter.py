from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
 chunk_size=3, chunk_overlap=0
)
document ="hii how are you"
texts = text_splitter.split_text(document)
print(texts)