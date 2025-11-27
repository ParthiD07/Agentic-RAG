from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
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

class GradeAnswer(BaseModel):
    "Binary score for the generated answer."
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'")

# 1. Create the parser from your Pydantic model
parser = JsonOutputParser(pydantic_object=GradeAnswer)

# 2. Update the system prompt to use the parser's instructions
system = """You are a grader assessing whether LLM generation successfully addresses and resolves the user's question \n .
    Your task is to compare the **generation answer** directly against the **user question**. If the generation fully answers the question, output 'yes'. Otherwise, output 'no'
    
    You must provide your answer in the following JSON format:
    {format_instructions}
    
    Do not output any other text or explanations."""

# 3. Inject the format_instructions into the prompt
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# 4. Build the chain with the prompt, LLM, and parser
answer_grader: RunnableSequence = answer_prompt | llm | parser
