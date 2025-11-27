from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs_per_url = [WebBaseLoader(url).load() for url in urls]

# Flatten into a single list of Document objects
docs_list = [doc for sub in docs_per_url for doc in sub]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,chunk_overlap=50)

docs_split = text_splitter.split_documents(docs_list)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # change to "cuda" if you have a GPU
)

vectorstore = Chroma.from_documents(
    documents=docs_split, collection_name="rag-chroma", 
    embedding= embeddings,persist_directory="./.chroma")

retriever = Chroma(
   collection_name="rag-chroma",
    embedding_function=embeddings,
    persist_directory="./.chroma").as_retriever()
