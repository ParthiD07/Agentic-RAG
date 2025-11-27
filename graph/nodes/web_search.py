from typing import Any,Dict
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

from dotenv import load_dotenv
load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Perform a web search using the question in the state and update the state with the results.

    Args:
        state (GraphState): The current graph state containing the question.

    Returns:
        Dict[str, Any]: Updated state with search results and web_search flag set to False.
    """

    print("---PERFORMING WEB SEARCH---")
    question = state['question']
    documents = state.get("documents", [])

    # Call Tavily and handle the dict-based response
    tavily_results = web_search_tool.invoke({"query":question})

    # Extract results safely
    results_list = tavily_results.get("results", [])
    if not results_list:
        print("No results found from Tavily.")
        return {"documents": documents, "question": question}

    # Join all result contents into a single text block
    joined_tavily_results= "\n".join(
        [
        f"Title: {result.get('title', 'No Title')}\nContent: {result.get('content', '')}\nURL: {result.get('url', '')}"
        for result in results_list])
    
    # Create a LangChain Document object
    web_results = Document(page_content=joined_tavily_results)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}

