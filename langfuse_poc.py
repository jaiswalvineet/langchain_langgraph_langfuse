from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
from langfuse.callback import CallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

    
def create_handler(user_identifier, trace_name):
    langfuse_handler = CallbackHandler(
        session_id=user_identifier,
        metadata={"a": "b"},
        trace_name=trace_name,
    )
    return langfuse_handler

def classification_agent(state):
    print(f"inside classification_agent")
    messages = state.get("messages")
    question = messages[-1]
    print(f"inside classification_agent {question}")
    llm = ChatOpenAI(temperature=0.7)
    langfuse_handler = create_handler("vineet", "classification_agent")
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "you are an AI bot who can classify"),
                ("user", "classify input either in sports_agent if related to sports or non_sports_agent if not related to sports category, input: {question}"),
            ]
        )
    retrieval_chain = (
        {"question": RunnablePassthrough()} | prompt | llm
    )
    output_data = retrieval_chain.invoke(
        {"question": question},
        config={
            "callbacks": [langfuse_handler],
        },
    )
    print(f"classification_agent output: {output_data}")
    return {"messages": [output_data]}

def sports_agent(state):
    messages = state.get("messages")
    question = messages[0]
    llm = ChatOpenAI(temperature=0.7)
    print(f"inside sports_agent {question}")
    langfuse_handler = create_handler("vineet", "sports_agent")
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "you are an AI bot who can generate more details"),
                ("user", "give me more detail about given sports  input: {question}"),
            ]
        )
    retrieval_chain = (
        {"question": RunnablePassthrough()} | prompt | llm
    )
    output_data = retrieval_chain.invoke(
        {"question": question},
        config={
            "callbacks": [langfuse_handler],
        },
    )
    print(f"sports_agent output {output_data}")
    return {"messages": [output_data]}

def non_sports_agent(state):
    messages = state.get("messages")
    question = messages[0]
    llm = ChatOpenAI(temperature=0.7)
    print(f"inside non_sports_agent {non_sports_agent}")
    langfuse_handler = create_handler("vineet", "non_sports_agent")
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "you are an AI bot who can generate more details"),
                ("user", "give me more detail about given input: {question}"),
            ]
        )
    retrieval_chain = (
       
        {"question": RunnablePassthrough()} | prompt | llm
    )
    output_data = retrieval_chain.invoke(
        {"question": question},
        config={
            "callbacks": [langfuse_handler],
        },
    )
    print(f"non_sports_agent output {output_data}")
    return {"messages": [output_data]}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
def _router(state):
    messages = state["messages"]
    last_message = messages[-1]
    return last_message

def execture_graph(input):
    inputs = {"messages": [input]}
    graph = StateGraph(AgentState)
    graph.add_node("classification_agent", classification_agent)
    graph.add_node("sports_agent", sports_agent)
    graph.add_node("non_sports_agent", non_sports_agent)
    graph.set_entry_point("classification_agent")
    graph.add_conditional_edges(
                "classification_agent",
                _router,
                {
                    "sports_agent": "sports_agent",
                    "non_sports_agent": "non_sports_agent"
                },
            )
    graph.add_edge("sports_agent", END)
    graph.add_edge("non_sports_agent", END)
    app = graph.compile()
    output_data = app.invoke(inputs)
    final_success_message = output_data.get("messages")[-1]
    return final_success_message

output = execture_graph("football")
print(f"final outout ==> \n\n  {output}")
