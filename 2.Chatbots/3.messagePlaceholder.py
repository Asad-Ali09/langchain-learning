from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Define the template structure
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful Customer Support Agent."),
    # MessagesPlaceholder specifies where the history will be inserted
    MessagesPlaceholder(variable_name="chat_placeholder"),
    ('human', '{query}')
])

# The list of messages representing the prior conversation
chat_history = [
    HumanMessage(content='I want to request a refund of order #1234'),
    AIMessage(content="Your refund request for order #1234 has been initialized")
]

prompt = chat_template.invoke({
    "chat_placeholder": chat_history, # <-- The list of history messages is passed here
    "query": 'Where is my refund?' # <-- The current query is passed here
})

print(prompt.to_string())

#Output
# messages=[HumanMessage(content='system', additional_kwargs={}, response_metadata={}), HumanMessage(content='You are a helpful Customer Support Agent.', additional_kwargs={}, response_metadata={}), HumanMessage(content='I want to request a refund of order #1234', additional_kwargs={}, response_metadata={}), AIMessage(content='Your refund request for order #1234 has been initialized', additional_kwargs={}, response_metadata={}), HumanMessage(content='Where is my refund?', additional_kwargs={}, response_metadata={})]