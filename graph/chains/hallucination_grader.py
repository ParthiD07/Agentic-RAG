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

class GradeHallucinations(BaseModel):
    "Binary score for hallucination present in the generation answer."
    binary_score: str = Field(
        description="Answer grounded in the facts, 'yes' or 'no'")

# 1. Create the parser from your Pydantic model
parser = JsonOutputParser(pydantic_object=GradeHallucinations)

# 2. Update the system prompt to use the parser's instructions
system = """You are a grader assessing whether an LLM generation is grounded in/ supported by a set of retrieved facts \n .
    Compare the GENERATION against the FACTS. If the GENERATION asserts something not present in the FACTS, the score is 'no'.
    
    You must provide your answer in the following JSON format:
    {format_instructions}
    
    Do not output any other text or explanations."""
# 3. Inject the format_instructions into the prompt
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# 4. Build the chain with the prompt, LLM, and parser
hallucination_grader: RunnableSequence = hallucination_prompt | llm | parser
