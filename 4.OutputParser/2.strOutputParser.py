from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define the prompt templates
template1 = PromptTemplate(template="Write a detailed report on topic {topic}", input_variables=["topic"])
template2 = PromptTemplate(template="Summarize the following text in 4 points: {text}", input_variables=["text"])

# Define the output parser
parser = StrOutputParser()

# Create a chain to automate the process
chain = template1 | model | parser | template2 | model | parser

# Invoke the chain
results = chain.invoke({"topic": "The impact of AI on modern education"})

print(results)