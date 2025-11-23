from langchain_core.runnables import RunnableParallel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='generate a short notes on {text} .',
    input_variables=['text']
)

prompt1 = PromptTemplate(
    template='give me five quiz questions based on text provided \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}.',
    input_variables=['notes', 'quiz']
)
llm1 = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct'
    , task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-1B-Instruct'
    , task="text-generation"
)

model = ChatHuggingFace(llm=llm1)
model1 = ChatHuggingFace(llm=llm2)

parser = StrOutputParser()

parallelchain = RunnableParallel({
    'notes':prompt | model | parser,
    'quiz':prompt1 | model1| parser
})

mergechain = prompt2 | model | parser

chain = parallelchain | mergechain

text = """
Large language models (LLMs) have transformed natural language processing with capabilities such as text generation, summarization, and question answering. While techniques such as retrieval-augmented generation (RAG) dynamically fetch external knowledge, they often introduce higher latency and system complexity.

Cache-augmented generation (CAG) offers an alternative by using expanded context windows and enhanced processing power in modern LLMs. By embedding and reusing precomputed knowledge within the model’s operational context, CAG enables faster and more efficient performance for static, knowledge-intensive tasks.

This article explores CAG and its integration with Granite language models, demonstrating how Granite’s extended context windows and processing power enhance efficiency by directly utilizing precomputed information.

How cache-augmented generation (CAG) works

Modern large language models (LLMs) can process up to 128K tokens—equivalent to 90-100 pages of an English document—without needing chunking or retrieval. Typically, the attention layer computes key-value (KV) representations for all knowledge with each query. Cache-augmented generation (CAG) optimizes this by precomputing KV representations once and reusing them, reducing redundant computations. This enhances retrieval efficiency and speeds up question-answering processes.

Preloading external knowledge

Relevant documents or datasets are preprocessed and loaded into the model’s extended context window.

Goal: Consolidate knowledge for answering queries.
Process:
Curate a static dataset.
Tokenize and format it for the model’s extended context.
Inject the dataset into the model’s inference pipeline.
Precomputing the key-value (KV) cache

The model processes the preloaded knowledge to generate a KV cache, storing intermediate states used in attention mechanisms.

Goal: Minimize redundant computations by storing reusable context.
Process: Encode documents into a KV cache using the model’s encoder, capturing its understanding of the preloaded knowledge.
Storing the KV cache

The precomputed KV cache is saved in memory or on disk for later use.

Goal: Enable multiple queries to access the cached knowledge without recomputation.
Benefit: The cache is computed once, allowing for rapid reuse during inference.
Inference with cached context

During inference, the model loads the cached context alongside user queries to generate responses.

Goal: Eliminate real-time retrieval and maintain contextual relevance.
Process:
Combine cached knowledge with the query.
Generate responses using the preloaded KV cache for efficiency and accuracy.
Cache reset (optional)

To optimize memory usage, the KV cache can be reset when needed.

Goal: Prevent overflow and manage memory efficiently.
Process: Remove unnecessary tokens or truncate the cache. Reinitialize the cache for new inference sessions.
Implementing CAG with Granite models

      """
result = chain.invoke({'text': text})
print(result)
chain.get_graph().print_ascii()