from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import OutputFixingParser, StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

schema = [
    ResponseSchema(name="fact1", description="first fact about black hole."),
    ResponseSchema(name="fact2", description="second fact about black hole."),
    ResponseSchema(name="fact3", description="third fact about black hole."),
]

parser = StructuredOutputParser.from_response_schemas(schema)
safe_parser = OutputFixingParser.from_llm(llm=model, parser=parser)

template = PromptTemplate(
    template="Provide 3 interesting facts about the {topic} in the following format:\n{{format_instructions}}\n",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | safe_parser

results = chain.invoke({"topic": "black holes"})

print(results)


