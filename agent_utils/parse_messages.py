import json
from typing import List, Dict, Any, Optional, Union
from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    ToolMessage, 
    SystemMessage,
    message_to_dict,
    messages_to_dict,
    convert_to_messages,
    filter_messages,
    get_buffer_string,
    trim_messages
)
from langchain_core.messages import BaseMessage, message_to_dict
from datetime import datetime
from langchain_core.messages.utils import AnyMessage

def parse_agent_messages(agent_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用LangChain原生API解析agent返回结果中的messages
    
    Args:
        agent_result: agent.invoke()的返回结果
        
    Returns:
        解析后的消息字典
    """
    messages: List[BaseMessage] = agent_result.get('messages', [])
    
    # 使用LangChain原生API转换为字典格式
    messages_dict = messages_to_dict(messages)
    
    parsed_result = {
        'raw_messages': messages,
        'messages_dict': messages_dict,
        'conversation_summary': {},
        'filtered_messages': {},
        'statistics': {}
    }
    
    # 按消息类型分类
    human_messages = filter_messages(messages, include_types=[HumanMessage])
    ai_messages = filter_messages(messages, include_types=[AIMessage])
    tool_messages = filter_messages(messages, include_types=[ToolMessage])
    system_messages = filter_messages(messages, include_types=[SystemMessage])
    
    parsed_result['filtered_messages'] = {
        'human': [message_to_dict(msg) for msg in human_messages],
        'ai': [message_to_dict(msg) for msg in ai_messages],
        'tool': [message_to_dict(msg) for msg in tool_messages],
        'system': [message_to_dict(msg) for msg in system_messages]
    }
    
    # 生成对话缓冲区字符串
    conversation_buffer = get_buffer_string(messages)
    parsed_result['conversation_summary']['buffer_string'] = conversation_buffer
    
    # 统计信息
    parsed_result['statistics'] = {
        'total_messages': len(messages),
        'human_count': len(human_messages),
        'ai_count': len(ai_messages),
        'tool_count': len(tool_messages),
        'system_count': len(system_messages)
    }
    
    # 提取token使用信息
    total_tokens = 0
    for ai_msg in ai_messages:
        if hasattr(ai_msg, 'usage_metadata') and ai_msg.usage_metadata:
            total_tokens += ai_msg.usage_metadata.get('total_tokens', 0)
    
    parsed_result['statistics']['total_tokens'] = total_tokens
    
    # 提取工具调用信息
    tool_calls_info = []
    for ai_msg in ai_messages:
        if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                tool_calls_info.append({
                    'name': tool_call.get('name'),
                    'args': tool_call.get('args', {}),
                    'id': tool_call.get('id')
                })
    
    parsed_result['tool_calls'] = tool_calls_info
    
    return parsed_result

def serialize_for_json(obj: Any) -> Any:
    """递归地将对象转换为JSON可序列化的格式"""
    if isinstance(obj, BaseMessage):
        return message_to_dict(obj)
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif hasattr(obj, '__dict__'):
        return serialize_for_json(obj.__dict__)
    else:
        return str(obj)

def print_json_pretty(data: Any, indent: int = 4, sort_keys: bool = True) -> None:
    """美化打印任何数据结构为JSON格式"""
    try:
        # 先序列化为JSON兼容格式
        serializable_data = serialize_for_json(data)
        
        # 美化输出
        formatted_json = json.dumps(
            serializable_data,
            indent=indent,
            ensure_ascii=False,  # 支持中文字符
            sort_keys=sort_keys,
            default=str  # 处理其他不可序列化的对象
        )
        print(formatted_json)
    except Exception as e:
        print(f"JSON序列化错误: {e}")
        print(f"原始数据类型: {type(data)}")
        print(f"原始数据: {data}")

def print_messages_only(agent_result: Dict) -> None:
    """只打印消息内容的简化版本"""
    try:
        parsed = parse_agent_messages(agent_result)
        # 只打印消息字典部分
        print_json_pretty(parsed['raw_messages'])
    except Exception as e:
        print(f"解析消息时出错: {e}")
        # 降级处理：直接转换消息
        messages = agent_result.get('messages', [])
        messages_dict = [message_to_dict(msg) for msg in messages]
        print_json_pretty(messages_dict)

def get_final_message(agent_result: Dict[str, Any]) -> str:
    """
    使用LangChain API提取最终答案
    
    Args:
        agent_result: agent.invoke()的返回结果
        
    Returns:
        最终的AI回答内容
    """
    messages: List[BaseMessage] = agent_result.get('messages', [])
    
    # 过滤出AI消息
    ai_messages = filter_messages(messages, include_types=[AIMessage])
    
    # 找到最后一个有内容的AI消息
    for ai_msg in reversed(ai_messages):
        if ai_msg.content and ai_msg.content.strip():
            return ai_msg.content
    
    return "No final answer found"

def parse_streaming_chunk(chunk: Any, stream_mode: str = "updates") -> Dict[str, Any]:
    """
    解析LangGraph流式响应中的单个chunk
    
    Args:
        chunk: 从agent.astream()或agent.stream()返回的单个chunk
        stream_mode: 流式模式 ("updates", "values", "messages", "custom")
        
    Returns:
        解析后的chunk信息
    """
    parsed_chunk = {
        'stream_mode': stream_mode,
        'chunk_type': type(chunk).__name__,
        'raw_chunk': chunk,
        'parsed_content': {},
        'tool_calls': [],
        'messages': [],
        'node_info': {}
    }
    
    try:
        if stream_mode == "updates":
            # updates模式下，chunk是字典，包含节点名作为key
            if isinstance(chunk, dict):
                for node_name, node_output in chunk.items():
                    parsed_chunk['node_info'][node_name] = {
                        'node_name': node_name,
                        'output_type': type(node_output).__name__
                    }
                    
                    # 处理包含messages的节点输出
                    if isinstance(node_output, dict) and 'messages' in node_output:
                        messages = node_output['messages']
                        parsed_chunk['messages'].extend(messages)
                        
                        # 提取工具调用信息
                        for msg in messages:
                            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    parsed_chunk['tool_calls'].append({
                                        'name': tool_call.get('name'),
                                        'args': tool_call.get('args', {}),
                                        'id': tool_call.get('id'),
                                        'type': tool_call.get('type', 'tool_call'),
                                        'node': node_name
                                    })
                            elif isinstance(msg, ToolMessage):
                                # 工具执行结果
                                parsed_chunk['parsed_content'][f'{node_name}_tool_result'] = {
                                    'tool_call_id': getattr(msg, 'tool_call_id', None),
                                    'name': getattr(msg, 'name', None),
                                    'content': msg.content
                                }
                    
                    parsed_chunk['parsed_content'][node_name] = node_output
                    
        elif stream_mode == "values":
            # values模式下，chunk包含完整的状态
            if isinstance(chunk, dict) and 'messages' in chunk:
                messages = chunk['messages']
                parsed_chunk['messages'] = messages
                
                # 提取最新的工具调用
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            parsed_chunk['tool_calls'].append({
                                'name': tool_call.get('name'),
                                'args': tool_call.get('args', {}),
                                'id': tool_call.get('id'),
                                'type': tool_call.get('type', 'tool_call')
                            })
                        break
                        
        elif stream_mode == "messages":
            # messages模式下，chunk是(token, metadata)元组
            if isinstance(chunk, tuple) and len(chunk) == 2:
                token, metadata = chunk
                parsed_chunk['parsed_content'] = {
                    'token': token,
                    'metadata': metadata,
                    'node': metadata.get('langgraph_node') if metadata else None
                }
                
        else:
            # 其他模式或自定义模式
            parsed_chunk['parsed_content'] = chunk
            
    except Exception as e:
        parsed_chunk['error'] = str(e)
        parsed_chunk['error_type'] = type(e).__name__
    
    return parsed_chunk

def extract_tool_calls_from_stream(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从流式响应的chunks中提取所有工具调用信息
    
    Args:
        chunks: 解析后的chunk列表
        
    Returns:
        工具调用信息列表
    """
    all_tool_calls = []
    tool_results = {}
    
    for chunk in chunks:
        # 收集工具调用
        if chunk.get('tool_calls'):
            all_tool_calls.extend(chunk['tool_calls'])
        
        # 收集工具执行结果
        parsed_content = chunk.get('parsed_content', {})
        for key, value in parsed_content.items():
            if key.endswith('_tool_result') and isinstance(value, dict):
                tool_call_id = value.get('tool_call_id')
                if tool_call_id:
                    tool_results[tool_call_id] = value
    
    # 将工具调用和结果关联
    for tool_call in all_tool_calls:
        tool_call_id = tool_call.get('id')
        if tool_call_id in tool_results:
            tool_call['result'] = tool_results[tool_call_id]
    
    return all_tool_calls

def parse_agent_streaming_response(agent, message: str, thread_id: str = "1", 
                                 stream_mode: str = "updates") -> Dict[str, Any]:
    """
    解析agent的完整流式响应
    
    Args:
        agent: LangGraph agent实例
        message: 用户消息
        thread_id: 线程ID
        stream_mode: 流式模式
        
    Returns:
        完整的解析结果
    """
    config = {"configurable": {"thread_id": thread_id}}
    chunks = []
    parsed_chunks = []
    
    try:
        # 收集所有chunks
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode=stream_mode
        ):
            chunks.append(chunk)
            parsed_chunk = parse_streaming_chunk(chunk, stream_mode)
            parsed_chunks.append(parsed_chunk)
    
    except Exception as e:
        return {
            'error': str(e),
            'error_type': type(e).__name__,
            'chunks': chunks,
            'parsed_chunks': parsed_chunks
        }
    
    # 提取工具调用信息
    tool_calls = extract_tool_calls_from_stream(parsed_chunks)
    
    # 提取最终消息
    final_messages = []
    for chunk in parsed_chunks:
        if chunk.get('messages'):
            final_messages.extend(chunk['messages'])
    
    # 去重并排序消息
    unique_messages = []
    seen_ids = set()
    for msg in final_messages:
        msg_id = getattr(msg, 'id', None) or id(msg)
        if msg_id not in seen_ids:
            unique_messages.append(msg)
            seen_ids.add(msg_id)
    
    return {
        'success': True,
        'stream_mode': stream_mode,
        'total_chunks': len(chunks),
        'raw_chunks': chunks,
        'parsed_chunks': parsed_chunks,
        'tool_calls': tool_calls,
        'final_messages': unique_messages,
        'conversation_summary': get_buffer_string(unique_messages) if unique_messages else "",
        'statistics': {
            'total_chunks': len(chunks),
            'tool_calls_count': len(tool_calls),
            'messages_count': len(unique_messages)
        }
    }

async def improved_stream_and_parse(agent, message: str, thread_id: str = "1", 
                                  stream_mode: str = "updates"):
    """
    改进的流式解析函数，支持实时解析和完整结果
    """
    print(f"\n=== Mode: {stream_mode} ===")
    
    # 方式1: 实时解析每个chunk
    config = {"configurable": {"thread_id": thread_id}}
    chunks_processed = 0
    
    try:
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode=stream_mode
        ):
            chunks_processed += 1
            parsed_chunk = parse_streaming_chunk(chunk, stream_mode)
            
            # print(f"\n--- Chunk {chunks_processed} ---")
            # print(f"节点信息: {parsed_chunk.get('node_info', {})}")
            
            # 显示工具调用
            if parsed_chunk.get('tool_calls'):
                print("🔧 Tool:")
                for tool_call in parsed_chunk['tool_calls']:
                    print(f"  - {tool_call['name']}({tool_call['args']})")
            
            # 显示工具结果
            parsed_content = parsed_chunk.get('parsed_content', {})
            for key, value in parsed_content.items():
                if key.endswith('_tool_result'):
                    print(f"🔧 Result of Tool: {value.get('name')} -> {value.get('content')}")
    
    except Exception as e:
        print(f"流式处理错误: {e}")
    
    # 方式2: 获取完整解析结果（同步版本）
    # print("\n=== 获取完整解析结果 ===")
    complete_result = parse_agent_streaming_response(agent, message, thread_id, stream_mode)
    
    if complete_result.get('success'):
        # print(f"总chunks: {complete_result['total_chunks']}")
        # print(f"工具调用数: {complete_result['statistics']['tool_calls_count']}")
        
        print(f"\n📝 Conversations:\n{complete_result['conversation_summary']}")
    else:
        print(f"解析失败: {complete_result.get('error')}")