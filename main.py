from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator

from dotenv import load_dotenv

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()

model = ChatOllama(
    model="qwen2.5-coder:1.5b"
)

class ResearcherState(TypedDict):
    messages: Annotated[list, add_messages]
    research_results: list
    node_history: Annotated[list[str], operator.add]

def researcher_node(state: ResearcherState):
    last_message = state["messages"][-1]

    wrapper = DuckDuckGoSearchAPIWrapper(max_results=1)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)

    
    results = search.run(last_message.content)
    return {"research_results": results, "node_history": ["researcher"]}


researcher_subgraph = (
    StateGraph(ResearcherState)
    .add_node("researcher", researcher_node)
    .add_edge(START, "researcher")
    .add_edge("researcher", END)
).compile()

class CoderResult(BaseModel):
    source_code: str = Field(
        ...,
        description = "Source code for the program"
    )

class CoderState(TypedDict):
    messages: Annotated[list, add_messages]
    node_history: Annotated[list[str], operator.add]
    research_results: list 
    source_code: str
    suggestion: str

def coder_node(state: CoderState):
    last_message = state["messages"][-1]

    coder_model = model.with_structured_output(CoderResult)

    SYSTEM_PROMPT = """You are an expert developer in writing programs.
                Never use any external libraries that requires installation from pip.
                Output just the source code for the program only.
                Write concise, effective code to minimize the length of the source code.
                The code should run strictly according to the user's requests.
                No personal tweaks and comments are allowed.
                Do only what the user requests and follow strictly the user's requirements.
    """
    
    if state.get("source_code", "") == "":
        result = coder_model.invoke([
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
                                Here are some relevant additional information (can be irrelevant), do not copy code from here: {state.get("research_results", "nothing here")}
                                {last_message.content}
    """
            }
        ])
    else:
        result = coder_model.invoke([
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
                                {last_message.content}
                                Here is your previous code for this request:
                                {state.get("source_code")}
                                Here are some suggestions:
                                {state.get("suggestion", "")}
                                Here are some relevant additional information (can be irrelevant), do not copy code from here: {state.get("research_results", "nothing here")}
                                
    """
            }
        ])

    return {"source_code": result.source_code, "node_history": ["coder"]}

coder_subgraph = (
    StateGraph(CoderState)
    .add_node("coder", coder_node)
    .add_edge(START, "coder")
    .add_edge("coder", END)
).compile()

class ValidationResults(BaseModel):
    approved: bool = Field(
        ...,
        description = "Whether the program satisfies the user's request or not."
    )
    suggestion: str = Field(
        ...,
        description = "Give suggestions to fix the program to satisfy the user's request."
    )

class QAState(TypedDict):
    messages: Annotated[list, add_messages]
    node_history: Annotated[list[str], operator.add]
    source_code: str
    approved: bool
    suggestion: str

def qa_node(state: QAState):
    last_message = state["messages"][-1]

    validation_model = model.with_structured_output(ValidationResults)

    result = validation_model.invoke([
        {
            "role": "system",
            "content": """
                            You are a expert at checking and validating programs.
                            Identify if the program satisfies the user's requests or not.
                            If the program satisfy the user's requests, approve it.
                            Provide straight-forward, concise suggestions to fix the program only when the program is not approved.
                            Program does not have to have features that the user didn't ask for.
                            Program just need to have features that the user asked for.
                            Don't be too strict on small details, unless the user explicitly requires attention of those small details.
                            Be strict with fundamental, important details.

                            **Example**: print("Hello, World!") can be approved, even when user requests print("Hello World"), because that detail is too small.
"""
        },
        {
            "role": "user",
            "content": f"Here is my request: {last_message.content}.\nHere is the source code of the program:\n" + state.get("source_code")
        }
    ])

    return {"approved": result.approved, "suggestion": result.suggestion, "node_history": ["qa"]}
    
    
    

qa_subgraph = (
    StateGraph(QAState)
    .add_node("qa", qa_node)
    .add_edge(START, "qa")
    .add_edge("qa", END)
).compile()

class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    node_history: Annotated[list[str], operator.add]
    research_results: list
    source_code: str
    approved: bool
    suggestion: str
    loop_count: int
    failed: bool

def supervisor_node(state: MainState):
    node_history = state.get("node_history", [])
    if node_history == []:
        return Command(goto="researcher", update={"loop_count": state["loop_count"]+1})
    last_node = node_history[-1]
    if last_node == "coder":
        return Command(goto="qa", update={"loop_count": state["loop_count"]+1})
    if state["loop_count"] > 20:
        return Command(goto="output", update={"failed": True})
    if not state.get("approved", False):
        return Command(goto="coder", update={"loop_count": state["loop_count"]+1})
    return Command(goto="output", update={"failed": False})
    

def output_node(state: MainState):
    if not state["failed"]:
        return {"messages": {"role": "assistant", "content": state["source_code"]}}
    return {"messages": {"role": "assistant", "content": "Sorry, we failed to code"}}

graph = (
    StateGraph(MainState)
    .add_node("supervisor", supervisor_node)
    .add_node("output", output_node)
    .add_node("researcher", researcher_subgraph)
    .add_node("coder", coder_subgraph)
    .add_node("qa", qa_subgraph)
    .add_edge(START, "supervisor")
    # .add_conditional_edges("supervisor",
    #                        lambda state: state.get("next"), 
    #                        {"researcher": "researcher", "coder": "coder", "qa": "qa"}
    # )
    .add_edge("researcher", "supervisor")
    .add_edge("coder", "supervisor")
    .add_edge("qa", "supervisor")
).compile()

def run_chatbot():

    while True:
        user_input = input("User (q to exit): ")
        if user_input == "q": break
        print("\n--------------------------")
        print("AI WORKING")
        print("\n--------------------------")
        for chunk in graph.stream(
            {"messages": {"role": "user", "content": user_input}, "loop_count": 0},
            stream_mode=["updates"],
            version="v2",
        ):
            if chunk["type"] != "updates":
                continue
            for node_name, state in chunk["data"].items():
                if node_name == "researcher":
                    print(f"RESEARCHER: {state["research_results"]}")
                elif node_name == "coder":
                    print(f"CODER:\n{state["source_code"]}")
                elif node_name == "qa":
                    print(f"QA:\n - Approved: {state["approved"]}\n - Suggestion: {state["suggestion"] if state["suggestion"] != "" else "nothing"}")
                elif node_name == "output":
                    print("------------------------\nAssistant:")
                    print(state["messages"]["content"])

if __name__ == "__main__":
    run_chatbot()