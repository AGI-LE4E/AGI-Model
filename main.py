from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from llm_utils import builder

load_dotenv(override=True)

if __name__ == "__main__":
    graph = builder.compile()
    human_message = HumanMessage(content="저는 HOTEL LEO 에서 숙박중입니다. 도민 매출금액 비율이 높은 식당을 추천해주세요.")
    res = graph.invoke(input=human_message)
    print('-------------------')
    print(res[-1].content)
    print('-------------------')

