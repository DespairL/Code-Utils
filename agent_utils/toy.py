import os
import json
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from parse_messages import print_messages_only, improved_stream_and_parse

load_dotenv()

API_KEY = os.getenv('API_KEY')
BASE_URL = os.getenv('BASE_URL')

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!" # mode = messages 一个一个字更新 / mode = updates 更新整个句子

from langgraph.config import get_stream_writer

def custom_get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    # stream any arbitrary data
    writer(f"Looking up data for city: {city}") # mode = custom 可以输出这个信息
    return f"It's always sunny in {city}!"

llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", base_url=BASE_URL, api_key=API_KEY, temperature=0.1) 
checkpointer = InMemorySaver()

agent_with_memory = create_react_agent(
    model=llm,
    tools=[custom_get_weather],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "1"}}

# for token, metadata in agent_with_memory.stream(
#     {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
#     config=config,
#     stream_mode="messages"
# ):
#     print("Token", token)
#     print("Metadata", metadata)
#     print("\n")

for chunk in agent_with_memory.stream(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config=config,
    stream_mode="custom"
):
    print(chunk)
    print("\n")