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

def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  
    user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", base_url=BASE_URL, api_key=API_KEY, temperature=0.1) 
checkpointer = InMemorySaver()

# use prompt
# agent = create_react_agent(
#     model=llm,
#     tools=[get_weather],
#     prompt=prompt,
# )

async def stream_agent_response(agent, message: str, thread_id: str = "1", stream_mode: str = "updates"):
    config = {"configurable": {"thread_id": thread_id}}
    
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": message}]},
        config,
        stream_mode=stream_mode
    ):
        yield chunk

async def receive_agent_response(agent, message: str, thread_id: str = "1", stream_mode: str = "updates"):
    async for chunk in stream_agent_response(
        agent, message, thread_id=thread_id, stream_mode=stream_mode
    ):
        print(chunk)

async def stream_and_parse(agent, message: str, thread_id: str = "1", stream_mode: str = "updates"):
    """改进的流式解析函数，包含错误处理和更好的chunk处理"""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode=stream_mode,
        ):
            print(f"\n=== Stream Mode: {stream_mode} ===")
            print(f"Chunk type: {type(chunk)}")
            print(f"Chunk content: {chunk}")
            
            # 根据不同的stream_mode处理chunk
            if stream_mode == "updates":
                # updates模式下，chunk是字典，包含节点名作为key
                for node_name, node_output in chunk.items():
                    print(f"\n--- Node: {node_name} ---")
                    if isinstance(node_output, dict) and 'messages' in node_output:
                        try:
                            parsed = parse_agent_messages(node_output)
                            print("Parsed messages:")
                            print_json_pretty(parsed['filtered_messages'])
                        except Exception as parse_error:
                            print(f"Parse error: {parse_error}")
                            print(f"Raw node output: {node_output}")
                    else:
                        print(f"Node output: {node_output}")
                        
            elif stream_mode == "values":
                # values模式下，chunk包含完整的状态
                if isinstance(chunk, dict) and 'messages' in chunk:
                    try:
                        parsed = parse_agent_messages(chunk)
                        print("Parsed state:")
                        print_json_pretty(parsed['filtered_messages'])
                    except Exception as parse_error:
                        print(f"Parse error: {parse_error}")
                        print(f"Raw chunk: {chunk}")
                        
            elif stream_mode == "messages":
                # messages模式下，chunk是(token, metadata)元组
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    token, metadata = chunk
                    print(f"Token: {token}")
                    print(f"Metadata: {metadata}")
                else:
                    print(f"Unexpected message chunk format: {chunk}")
                    
            else:
                # 其他模式，直接打印
                print(f"Raw chunk: {chunk}")
                
    except openai.InternalServerError as e:
        print(f"OpenAI服务器错误: {e}")
        print("请稍后重试，这通常是临时的服务器问题")
    except Exception as e:
        print(f"Stream处理错误: {e}")
        print(f"错误类型: {type(e)}")

# use memory
agent_with_memory = create_react_agent(
    model=llm,
    tools=[get_weather],
    checkpointer=checkpointer,
)

if __name__ == "__main__":
    # Synchronous - .invoke() or .stream()
    # Asynchronous - await .ainvoke() or async for with .astream()
    asyncio.run(improved_stream_and_parse(agent_with_memory, "what is the weather in sf?"))
    asyncio.run(improved_stream_and_parse(agent_with_memory, "which city I mentioned before?"))