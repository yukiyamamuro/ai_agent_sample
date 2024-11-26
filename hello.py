from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("こんにちは、私はポリンキーです"),
    AIMessage(content="こんにちは、ポリンキーさん。"),
    HumanMessage("私の名前はわかりますか?"),
]

ai_message = model.invoke(messages)
print(ai_message.content)
