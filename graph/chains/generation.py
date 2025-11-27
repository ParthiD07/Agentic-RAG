from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv
load_dotenv()

# Set up the Endpoint
endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0,
    huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'])

# Wrap it with ChatHuggingFace
llm = ChatHuggingFace(llm=endpoint)

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()

