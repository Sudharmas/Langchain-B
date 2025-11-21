from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()
embedd = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

doc = [
    "ice cream is cold",
    "tea is hot",
    "bangalore is silicon city of india with more than 1 million people living in single city"
]

query = 'tell me about ice cream'
result = embedd.embed_documents(doc)
qyeryembed = embedd.embed_query(query)

cosine = cosine_similarity(result,[qyeryembed])
index ,score = sorted(list(enumerate(cosine)) ,key=lambda x:x[1])[-1] #enumerate will add index to all embeds and list will convert the enumerated embeds into list of embedding,then i t will be sorted on basis of the second argument inside list bcz there will be 2 parameters in single list item,so we use lambda to tell that to sort based on 1 or first elemne tnot 0th element in list items.
print(query)
print(doc[index])
print(score)
