from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import (
    PydanticToolsParser,
)
from .schemas import Place
from .tools import query_restaurants

from dotenv import load_dotenv

load_dotenv(override=True)

# Extractor
parser_pydantic = PydanticToolsParser(tools=[Place])

llm = ChatOpenAI(model="gpt-4o")

extract_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                너는 사용자의 입력에서 사용자의 위치정보를 추출해내는 Assistance이다.
                사용자의 입력에 대해 사용자가 어디에 있는지 장소명을 추출해내는 것이 목표이다.
            """,
        ),
        MessagesPlaceholder(variable_name="search_query"),
        (
            "system",
            "위 유저 Input에 맞는 장소명을 추출해주시길 바랍니다.",
        ),
    ]
)

initial_extractor = extract_prompt | llm.bind_tools(
    tools=[Place], tool_choice="Place"
)

extract_chain = initial_extractor | parser_pydantic

# Searcher
search_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                너는 Extractor가 추출한 장소명을 기반으로 주변 맛집을 검색하는 Assistance이다.
                주어진 장소를 기반으로 어떤 tool을 사용해 주변 맛집을 검색할지 선택해야 한다.
            """,
        ),
        MessagesPlaceholder(variable_name="search_term"),
        (
            "system",
            "위의 장소를 기반으로 주변 음식점을 검색해주세요."
            "다음 tool을 사용할 수 있습니다:"
            "query_restaurants - 장소를 기반으로 음식점을 DB에서 쿼리할 수 있습니다."
            "아래 툴을 사용해주세요",
        ),
        MessagesPlaceholder(variable_name="tool_choice"),
    ]
)

tools = [query_restaurants]

search_chain = search_prompt | llm.bind_tools(tools)

recommend_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                너는 검색결과를 바탕으로 사용자에게 식당을 추천하는 Assistance이다.
                적절한 식당을 추천하는 것이 당신의 필수적인 역할입니다.
                응답은 아래 형태를 유지해야 합니다.
                최소 5개의 식당을 추천해주세요.
                
                **음식점 이름1 **
                음식점의 위치: 위치 정보
                **음식점 이름2 **
                음식점의 위치: 위치 정보
            """,
        ),
        MessagesPlaceholder(variable_name="user_input"),
        (
            "system",
            "위 유저의 요구사항을 만족하여, 아래 검색된 맛집 목록을 중에서 추천해주세요"
            # "유저의 요구사항을 만족하는 맛집이 없다면 '해당하는 맛집이 없습니다'라고 응답해주세요.",
        ),
        MessagesPlaceholder(variable_name="restaurants_list"),
    ]
)

recommend_chain = recommend_prompt | llm
