from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import promptTemplate

model = ChatOpenAI(model="gpt-4o-mini", temperature=1)

# ====== Define Pydantic Model ======
class Recipe(BaseModel):  # noqa: D101
    ingredients: list[str] = Field(description="材料")
    steps: list[str] = Field(description="手順")

output_parser = PydanticOutputParser(pydantic_object=Recipe)
format_instructions = output_parser.get_format_instructions()


# ====== Define Prompt ======
prompt = promptTemplate.recipe_prompt
prompt_with_format_instructions = prompt.partial(
    format_instructions=format_instructions
)

def printerer(text: str) -> str:
    print(text)
    return text

base_chain = (
    prompt_with_format_instructions
    | RunnableLambda(printerer)
    | model
    | output_parser
)

synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "レシピの似た点があれば教えてください"),
        ("human", "レシピ1: {rice}\n\nレシピ2: {nann}"),
    ]
)

synthesize_chain = (RunnableParallel({
    "rice": base_chain,
    "nann": base_chain
    })
    | synthesize_prompt
)

result=synthesize_chain.invoke({"dish": "チキンカレー"})
print("===result===")
print(result)

