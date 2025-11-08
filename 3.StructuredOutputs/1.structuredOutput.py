from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Schema
class Review(TypedDict):
    summary: str
    sentiment: str

strucutred_model = model.with_structured_output(Review)

prompt = """
The hardware is great, but the software feels kind of bloated. There are too many bloated apps. My phone hangs whenever I play bigger games like PUBG.
"""

results = strucutred_model.invoke(prompt)

prompt_for_unstructured_model = f'generate sentiment and summary in a json form of the given review. The review is {prompt}'

unstructured_results = model.invoke(prompt_for_unstructured_model)

print(results)
print("-----")
print(unstructured_results)
print("-----")
print(unstructured_results.content)