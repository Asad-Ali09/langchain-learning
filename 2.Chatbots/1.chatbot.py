from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower() == "exit":
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    
    print("Bot:", result.content)
    print("Token Usage: (in:", result.usage_metadata["input_tokens"], ") - (out:", result.usage_metadata["output_tokens"], ") - (total:",  result.usage_metadata["total_tokens"], ")")


print(chat_history)