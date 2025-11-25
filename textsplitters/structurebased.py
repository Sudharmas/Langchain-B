from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
 chunk_size=30, chunk_overlap=0
)
document ="hii how are you hii how are you hii how are youhii how are youhii how are youhii how are youhii how are youhii how are youhii how are youhii how are youhii how are youhii how are youhii how are youhii how are you"
texts = text_splitter.split_text(document)
print(len(texts))
