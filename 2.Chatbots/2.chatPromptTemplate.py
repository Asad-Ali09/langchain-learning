from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} export."),
    ('human', "Please provide information about {topic} in simple terms.")
])

prompt = chat_template.invoke({
    'domain': 'quantum physics',
    'topic': 'wormhole'
})

print(prompt)
