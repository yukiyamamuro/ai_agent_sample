import operator
from typing import Annotated

from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import Graph

ROLES = {
    "1": {
        "name": "一般知識エキスパート",
        "description": "幅広い分野の一般的な質問に答える",
        "details": "幅広い分野の一般的な質問に対して、正確でわかりやすい回答を提供してください。"  # noqa: E501
    },
    "2": {
        "name": "生成AI製品のエキスパート",
        "description": "生成AIや関連製品、技術に関する専門的な質問に答える",
        "details": "生成AIや関連製品、技術に関する専門的な質問に対して、最新の情報と深い洞察を提供してください。"  # noqa: E501
    },
    "3": {
        "name": "カウンセラー",
        "description": "個人的な悩みや心理的な問題に対してサポートを提供する",
        "details": "個人的な悩みや心理的な問題に対して、共感的で支援的な回答を提供し、可能であれば適切にアドバイスもおこなってください。"  # noqa: E501
    },
}


class State(BaseModel):
    query: str = Field(description="ユーザーからの質問")
    current_role: str = Field(
        default="",
        description="選定された回答ロール",
    )
    message: Annotated[list[str], operator.add] = Field(
        default=[], description="回答履歴"
    )
    current_judge: bool = Field(
        default=False, description="品質チェックの結果"
    )
    judgement_reason: str = Field(
        default="", description="品質チェックの判定理由"
    )

workflow = Graph(State)

