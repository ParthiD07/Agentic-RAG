from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
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

class GradeDocuments(BaseModel):
    "Binary score for relevance check on retrieved docuemnts"
    binary_score: str = Field(
        description="Documents are relevent to the question, 'yes' or 'no'")

# 1. Create the parser from your Pydantic model
parser = JsonOutputParser(pydantic_object=GradeDocuments)

# 2. Update the system prompt to use the parser's instructions
system = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as yes or if no relevent document found mention it as no.
    
    You must provide your answer in the following JSON format:
    {format_instructions}
    
    Do not output any other text or explanations."""

# 3. Inject the format_instructions into the prompt
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# 4. Build the chain with the prompt, LLM, and parser
retrieval_grader = grade_prompt | llm | parser

