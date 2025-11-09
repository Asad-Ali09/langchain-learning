from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Manual step-by-step invocation without using a chain
# Define the prompt templates
template1 = PromptTemplate(template="Write a detailed report on topic {topic}", input_variables=["topic"])
template2 = PromptTemplate(template="Summarize the following text in 4 points: {text}", input_variables=["text"])

# Create the first prompt and get the detailed report
prompt1 = template1.invoke({"topic": "The impact of AI on modern education"})
result1 = model.invoke(prompt1).content

# Create the second prompt and get the summary using the detailed report as input
prompt2 = template2.invoke({"text": result1})
result2 = model.invoke(prompt2).content


print("Detailed Report:\n", result1)
print("\n\n\n*********************************\n\n\n")
print("\nSummary in 4 points:\n", result2)
