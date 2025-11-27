from typing import Any,Dict
from graph.state import GraphState
from graph.chains.generation import generation_chain

def generate_answer(state: GraphState) -> Dict[str, Any]:
    """
    Generate an answer based on the context documents and question in the state.

    Args:
        state (GraphState): The current graph state containing documents and question.

    Returns:
        Dict[str, Any]: Updated state with the generated answer.
    """

    print("---GENERATING ANSWER FROM DOCUMENTS---")
    question = state['question']
    documents = state['documents']
    

    output = generation_chain.invoke({"context": documents, "question": question})

    return {"generation": output, "question": question, "documents": documents}