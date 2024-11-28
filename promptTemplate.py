from langchain_core.prompts import ChatPromptTemplate

cow_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたはアスキーアート職人です。\n\n以下の[#牛に言わせる言葉]の内容を使い、cowsayコマンドの出力のように牛にメッセージを言わせてください。",
        ),
        ("human", "#牛に言わせる言葉\n{body}"),
    ]
)

recipe_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザーが入力した料理のレシピを考えてください。\n\n"
            "{format_instructions}",
        ),
        ("human", "{dish}"),
    ]
)
