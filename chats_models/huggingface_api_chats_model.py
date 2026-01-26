from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512,      # Control response length
    temperature=0.7,          # Control randomness (0.0-1.0)
    top_p=0.95,              # Nucleus sampling
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of india")

print(result.content)