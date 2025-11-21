from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

st.header("reserach tool")

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct"
    ,task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# text = st.text_input("enter text") # ask user input directly

paper = st.selectbox("select the paper",["GenAI : the future","The Algorithmic disorder","RideOne:ride booking conflicts"])
style = st.selectbox("select the style",["Intuitive","Creative","Logical"])
length = st.selectbox("select the style",["short 1,2 para","medium 5-7 para","summary 10+ para"])


template = load_prompt('./template.json')


# prompt = template.invoke({
#     'length_input': length,
#     'paper_input': paper,
#     'style_input': style,
# })
# instead of using 2 invokes ,using concept of chain we can invoke only once for both of them

if st.button("submit"):
    chain = template | model
    result = chain.invoke({
    'length_input': length,
    'paper_input': paper,
    'style_input': style,
})
    st.write(result.content)
