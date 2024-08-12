from dotenv import load_dotenv

from typing import Sequence
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.prebuilt import ToolNode

from langgraph.graph import END, MessageGraph

from .chains import extract_chain, search_chain, recommend_chain
from llm_utils.tools import query_restaurants

tool_node = ToolNode([query_restaurants])

EXTRACT = "extract"
SEARCH = "search"
VALIDATE = "validate"
RECOMMEND = "recommend"
SEARCH_TOOL = "search_tool"

def extract_node(state: Sequence[BaseMessage]):
    res = extract_chain.invoke(input={"search_query": [state[-1].content]})

    search_query = " ".join(list(res[0].dict().values()))
    if "분류불가" in search_query:
        return "잘 이해하지 못했어요. 다시 말씀해주세요."
    else:
        return search_query
    

def search_node(state: Sequence[BaseMessage]):
    search_term = state[-1].content
    tool_choice = "query_restaurants"

    res = search_chain.invoke(input={"search_term": [search_term], "tool_choice": [tool_choice]})

    return res

def recommend_node(state: Sequence[BaseMessage]):
    res = recommend_chain.invoke(
        input={
            "user_input": [state[0].content],
            "restaurants_list": [state[-1].content],
        }
    )

    return res


builder = MessageGraph()

# Node
builder.add_node(EXTRACT, extract_node)
builder.add_node(SEARCH, search_node)
builder.add_node(RECOMMEND, recommend_node)
builder.add_node(SEARCH_TOOL, tool_node)

# Edge
builder.set_entry_point(EXTRACT)

builder.add_edge(EXTRACT, SEARCH)
builder.add_edge(SEARCH, SEARCH_TOOL)
builder.add_edge(SEARCH_TOOL, RECOMMEND)
builder.add_edge(RECOMMEND, END)

__all__ = ["builder"]