from dotenv import load_dotenv
load_dotenv()
from pprint import pprint
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.generation import generation_chain
from graph.chains.router import question_router, RouteQuery
from ingestion import retriever



def test_retriever_grader_answer_yes()->None:
    question="What is prompt engineering?"
    docs=retriever.invoke(question)
    first_doc=docs[0].page_content

    result : GradeDocuments = retrieval_grader.invoke(
        {"document":first_doc,"question":question}
    )
    assert result['binary_score'].lower()=="yes"

def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizza", "document": doc_txt}
    )

    assert res['binary_score'].lower() == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    output = generation_chain.invoke({"context": docs, "question": question})
    pprint(output)

def test_hallucination_grader_yes() -> None:
    question = "What is prompt engineering?"
    docs = retriever.invoke(question)
    output = generation_chain.invoke({"context": docs, "question": question})

    result = hallucination_grader.invoke(
        {"documents":docs,"generation":output})

    assert result['binary_score'].lower() == "yes"

def test_hallucination_grader_no() -> None:
    question = "What is prompt engineering?"
    docs = retriever.invoke(question)

    result = hallucination_grader.invoke(
        {"documents":docs,"generation":"In order to make pizza we need to first start with the dough"})

    assert result['binary_score'].lower() == "no"

def test_router_to_vectorstore() -> None:
    question = "What is prompt engineering?"
    result: RouteQuery = question_router.invoke({"question": question})
    assert result['datasource'] == "vectorstore"

def test_router_to_websearch() -> None:
    question = "how to make pizza?"
    result: RouteQuery = question_router.invoke({"question": question})
    assert result['datasource'] == "websearch"
