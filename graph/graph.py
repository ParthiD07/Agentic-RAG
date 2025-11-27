from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, StateGraph

from graph.consts import RETRIEVE, GENERATE_ANSWER, GRADE_DOCUMENTS, WEB_SEARCH
from graph.nodes import retrieve, generate_answer, grade_documents, web_search
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from graph.chains.router import RouteQuery,question_router
from graph.state import GraphState


def decide_to_generate(state) -> str:
    print("---ASSESS GRADE DOCUMENTS---")
    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return WEB_SEARCH
    else:
        print("---DECISION: ALL DOCUMENTS ARE RELEVANT TO QUESTION, GENERATE ANSWER---")
        return GENERATE_ANSWER

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score["binary_score"]:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke(
            { "question": question, "generation": generation}
        )
        if answer_grade := score["binary_score"]:
            print("---DECISION: GENERATION ANSWERS THE QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ANSWER THE QUESTION---")
            return "not_useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not_supported"
    
#---Corrective_RAG_Workflow---
workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE_ANSWER, generate_answer)

workflow.set_entry_point(RETRIEVE)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate,path_map={
    WEB_SEARCH: WEB_SEARCH,
    GENERATE_ANSWER: GENERATE_ANSWER})

workflow.add_edge(WEB_SEARCH, GENERATE_ANSWER)
workflow.add_edge(GENERATE_ANSWER, END)

#---Self_RAG_Worflow---
workflow1 = StateGraph(GraphState)
workflow1.add_node(RETRIEVE, retrieve)
workflow1.add_node(GRADE_DOCUMENTS, grade_documents)
workflow1.add_node(WEB_SEARCH, web_search)
workflow1.add_node(GENERATE_ANSWER, generate_answer)

workflow1.set_entry_point(RETRIEVE)

workflow1.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow1.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate,path_map={
    WEB_SEARCH: WEB_SEARCH,
    GENERATE_ANSWER: GENERATE_ANSWER})
workflow1.add_conditional_edges(GENERATE_ANSWER, grade_generation_grounded_in_documents_and_question,path_map={
    "not_supported": GENERATE_ANSWER,
    "useful": END,
    "not_useful": WEB_SEARCH})

workflow1.add_edge(WEB_SEARCH, GENERATE_ANSWER)
workflow1.add_edge(GENERATE_ANSWER, END)


#---Adpative_RAG_Worflow---

def route_question(state: GraphState)-> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question":question})
    if source["datasource"] == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEB_SEARCH
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION to RAG---")
        return RETRIEVE
    
workflow2 = StateGraph(GraphState)
workflow2.add_node(RETRIEVE, retrieve)
workflow2.add_node(GRADE_DOCUMENTS, grade_documents)
workflow2.add_node(WEB_SEARCH, web_search)
workflow2.add_node(GENERATE_ANSWER, generate_answer)

workflow2.set_conditional_entry_point(route_question,path_map={
    WEB_SEARCH: WEB_SEARCH,
    RETRIEVE: RETRIEVE})

workflow2.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow2.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate,path_map={
    WEB_SEARCH: WEB_SEARCH,
    GENERATE_ANSWER: GENERATE_ANSWER})
workflow2.add_conditional_edges(GENERATE_ANSWER, grade_generation_grounded_in_documents_and_question,path_map={
    "not_supported": GENERATE_ANSWER,
    "useful": END,
    "not_useful": WEB_SEARCH})

workflow2.add_edge(WEB_SEARCH, GENERATE_ANSWER)
workflow2.add_edge(GENERATE_ANSWER, END)

app= workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="corrective_RAG_workflow.png")

app1= workflow1.compile()
app1.get_graph().draw_mermaid_png(output_file_path="self_RAG_workflow.png")

app2= workflow2.compile()
app2.get_graph().draw_mermaid_png(output_file_path="adpative_RAG_workflow.png")