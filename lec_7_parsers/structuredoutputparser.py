import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq # Swapped for Groq
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Define the Groq model
# Make sure GROQ_API_KEY is in your .env file
model = ChatGroq(
    model_name="llama-3.3-70b-versatile", # High performance Groq model
    temperature=0.5,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

# --- Logic remains exactly as you wrote it ---

# 1. Generate the detailed report
prompt1 = template1.invoke({'topic':'black hole'})
result = model.invoke(prompt1)

# 2. Use the result to generate the summary
prompt2 = template2.invoke({'text': result.content})
result1 = model.invoke(prompt2)

print(result1.content)