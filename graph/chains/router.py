from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence
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

class RouteQuery(BaseModel):
    "Route a user query to the most relevant datasource"

    datasource : Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to vectorstore or websearch"
    )

# 1. Create the parser from your Pydantic model
parser = JsonOutputParser(pydantic_object=RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search.

You MUST provide your output exclusively in the following JSON format:
{format_instructions}

Do not output any other text, explanation, or conversational filer outside of the required JSON structure.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


question_router: RunnableSequence = route_prompt | llm | parser
