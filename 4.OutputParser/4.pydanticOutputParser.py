from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()


class AnimalFacts(BaseModel):
    name: str = Field(description="Name of the animal")
    habitat: str = Field(description="Where this animal usually lives")
    interesting_fact: str = Field(description="A fun or surprising fact about this animal")

parser = PydanticOutputParser(pydantic_object=AnimalFacts)

prompt = PromptTemplate(
    template="Provide information about {animal} in the following format:\n{format_instructions}\n",
    input_variables=["animal"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chain = prompt | model | parser

results = chain.invoke({"animal": "dolphin"})

print(results)