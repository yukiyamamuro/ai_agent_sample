from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_community import GoogleDriveLoader
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Recipe(BaseModel):  # noqa: D101
    ingredients: list[str] = Field(description="材料")
    steps: list[str] = Field(description="手順")

output_parser = PydanticOutputParser(pydantic_object=Recipe)
format_instructions = output_parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザーが入力した料理のレシピを考えてください。\n\n"
            "{format_instructions}",
        ),
        ("human", "{dish}"),
    ]
)

prompt_with_format_instructions = prompt.partial(
    format_instructions=format_instructions
)

chain = prompt_with_format_instructions | model | output_parser
recipe = chain.invoke({"dish":"カレー"})
print(recipe)

