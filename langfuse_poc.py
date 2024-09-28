from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
from langfuse.callback import CallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.decorators import observe, langfuse_context
import argparse


def create_handler(user_identifier, trace_name):
    langfuse_handler = CallbackHandler(
        session_id=user_identifier,
        metadata={"a": "b"},
        trace_name=trace_name,
    )
    return langfuse_handler


@observe()
def classification_agent(state):
    messages = state.get("messages")
    question = messages[-1]
    print(f"inside classification_agent {question}")
    llm = ChatOpenAI(temperature=0.7)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are an AI bot who can classify"),
            (
                "user",
                "classify input either in sports_agent if related to sports or non_sports_agent if not related to sports category, input: {question}",
            ),
        ]
    )
    output_data = classification_agent_generation(llm, prompt, question)
    print(f"classification_agent output: {output_data}")
    return {"messages": [output_data]}


@observe(as_type="generation")
def classification_agent_generation(llm, prompt, question):
    langfuse_handler = create_handler("vineet", "classification_agent")
    retrieval_chain = {"question": RunnablePassthrough()} | prompt | llm
    output_data = retrieval_chain.invoke(
        {"question": question},
        config={
            "callbacks": [langfuse_handler],
        },
    )
    return output_data


@observe()
def sports_agent(state):
    messages = state.get("messages")
    question = messages[0]
    llm = ChatOpenAI(temperature=0.7)
    print(f"inside sports_agent {question}")
    langfuse_handler = create_handler("vineet", "sports_agent")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are an AI bot who can generate more details"),
            ("user", "Give me top 10 world cup winner team in json format  for : {question}"),
        ]
    )
    output_data = sports_agent_generation(llm, prompt, question, langfuse_handler)

    return {"messages": [output_data]}


@observe(as_type="generation")
def sports_agent_generation(llm, prompt, question, langfuse_handler):
    retrieval_chain = {"question": RunnablePassthrough()} | prompt | llm
    output_data = retrieval_chain.invoke(
        {"question": question},
        config={
            "callbacks": [langfuse_handler],
        },
    )
    return output_data


@observe()
def non_sports_agent(state):
    messages = state.get("messages")
    question = messages[0]
    llm = ChatOpenAI(temperature=0.7)
    print(f"inside non_sports_agent {non_sports_agent}")
    langfuse_handler = create_handler("vineet", "non_sports_agent")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are an AI bot who can generate more details"),
            ("user", "give me more detail from wikipedia for : {question}"),
        ]
    )
    output_data = non_sports_agent_generation(llm, prompt, question, langfuse_handler)

    return {"messages": [output_data]}


@observe(as_type="generation")
def non_sports_agent_generation(llm, prompt, question, langfuse_handler):
    retrieval_chain = {"question": RunnablePassthrough()} | prompt | llm
    output_data = retrieval_chain.invoke(
        {"question": question},
        config={
            "callbacks": [langfuse_handler],
        },
    )

    return output_data


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def _router(state):
    messages = state["messages"]
    last_message = messages[-1]
    return last_message.content


@observe()
def execute_graph(input):
    inputs = {"messages": [input]}
    graph = StateGraph(AgentState)
    graph.add_node("classification_agent", classification_agent)
    graph.add_node("sports_agent", sports_agent)
    graph.add_node("non_sports_agent", non_sports_agent)
    graph.set_entry_point("classification_agent")
    graph.add_conditional_edges(
        "classification_agent",
        _router,
        {"sports_agent": "sports_agent", "non_sports_agent": "non_sports_agent"},
    )
    graph.add_edge("sports_agent", END)
    graph.add_edge("non_sports_agent", END)
    app = graph.compile()
    output_data = app.invoke(inputs)
    final_success_message = output_data.get("messages")[-1]
    return final_success_message


def main():
    parser = argparse.ArgumentParser(description="give your input ")
    parser.add_argument("--input", type=str, required=True, help="user input")
    args = parser.parse_args()
    result = execute_graph(args.input)
    print(result)


if __name__ == "__main__":
    main()
