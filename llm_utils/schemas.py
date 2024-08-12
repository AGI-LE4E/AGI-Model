from langchain_core.pydantic_v1 import BaseModel, Field


class Place(BaseModel):
    """Place"""

    place: str = Field(
        description="대화에서 나온 확실한 장소 필드. 대화에서 장소가 확실히 지정되지 않은 경우, '분류불가' 를 사용한다."
    )
