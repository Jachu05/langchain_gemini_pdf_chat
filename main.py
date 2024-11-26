from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1)

message = llm.invoke("""question: how are you
                     answer:
                     """)

print(message.content)
