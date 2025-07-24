import os
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any, Union

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_API_KEY = os.getenv('API_KEY')
DEFAULT_BASE_URL = os.getenv('BASE_URL', 'https://api.openai-proxy.org/v1')
DEFAULT_CLAUDE_KEY = os.getenv('CLAUDE_API_KEY')
DEFAULT_GEMINI_KEY = os.getenv('GEMINI_API_KEY')

def call_llm(
    model: str,
    messages: List[Dict[str, str]] = None,
    query: str = None,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    api_key: str = None,
    base_url: str = None,
    **kwargs
) -> str:
    """
    Unified LLM calling function that automatically selects the correct API based on model type
    
    Args:
        model: Model name, e.g. 'gpt-4', 'claude-sonnet-4-20250514', 'gemini-2.5-flash', 'deepseek-reasoner'
        messages: Message list in format [{'role': 'user', 'content': '...'}]
        query: Simple query string (will be converted to messages format if provided)
        system_prompt: System prompt
        max_tokens: Maximum token count
        temperature: Temperature parameter
        api_key: API key (optional, defaults to environment variable)
        base_url: API base URL (optional, defaults to environment variable)
        **kwargs: Other model-specific parameters
    
    Returns:
        str: Model response text
    """
    
    # Process input messages
    if messages is None and query is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    elif messages is None:
        raise ValueError("Must provide either messages or query parameter")
    
    # Auto-select calling method based on model name
    model_lower = model.lower()
    
    # OpenAI models (including GPT series and other OpenAI-compatible models)
    if any(x in model_lower for x in ['gpt', 'openai', 'deepseek', 'o1']):
        return _call_openai_compatible(model, messages, max_tokens, temperature, api_key, base_url, **kwargs)
    
    # Claude models
    elif any(x in model_lower for x in ['claude', 'anthropic', 'sonnet', 'haiku', 'opus']):
        return _call_claude(model, messages, system_prompt, max_tokens, api_key, **kwargs)
    
    # Gemini models
    elif any(x in model_lower for x in ['gemini', 'google']):
        return _call_gemini(model, messages, system_prompt, api_key, **kwargs)
    
    else:
        # Default to OpenAI-compatible interface
        return _call_openai_compatible(model, messages, max_tokens, temperature, api_key, base_url, **kwargs)

def _call_openai_compatible(
    model: str, 
    messages: List[Dict[str, str]], 
    max_tokens: int, 
    temperature: float,
    api_key: str = None, 
    base_url: str = None,
    **kwargs
) -> str:
    """Call OpenAI-compatible API (including GPT, DeepSeek, etc.)"""
    api_key = api_key or DEFAULT_API_KEY
    base_url = base_url or DEFAULT_BASE_URL
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"OpenAI API error: {str(e)}"

def _call_claude(
    model: str, 
    messages: List[Dict[str, str]], 
    system_prompt: str,
    max_tokens: int,
    api_key: str = None,
    **kwargs
) -> str:
    """Call Claude API"""
    api_key = api_key or DEFAULT_CLAUDE_KEY
    
    client = Anthropic(
        base_url='https://api.openai-proxy.org/anthropic',
        api_key=api_key
    )
    
    try:
        # Filter out system messages, Claude handles system prompt separately
        claude_messages = [msg for msg in messages if msg['role'] != 'system']
        
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=claude_messages,
            **kwargs
        )
        
        # Extract text content
        if hasattr(message.content[0], 'text'):
            return message.content[0].text
        else:
            return str(message.content[0])
            
    except Exception as e:
        return f"Claude API error: {str(e)}"

def _call_gemini(
    model: str, 
    messages: List[Dict[str, str]], 
    system_prompt: str,
    api_key: str = None,
    **kwargs
) -> str:
    """Call Gemini API"""
    api_key = api_key or DEFAULT_GEMINI_KEY
    
    try:
        client = genai.Client(
            api_key=api_key,
            http_options={
                "base_url": "https://api.openai-proxy.org/google"
            }
        )
        
        # Convert message format
        filtered_messages = [msg for msg in messages if msg['role'] != 'system']
        
        # Simplified handling: use string content directly for single user message
        if len(filtered_messages) == 1 and filtered_messages[0]['role'] == 'user':
            content_input = filtered_messages[0]['content']
        else:
            # For multi-turn conversations, build correct content format
            content_input = []
            for msg in filtered_messages:
                if msg['role'] == 'user':
                    content_input.append(types.Content(
                        role='user',
                        parts=[types.Part(text=msg['content'])]
                    ))
                elif msg['role'] == 'assistant':
                    content_input.append(types.Content(
                        role='model',
                        parts=[types.Part(text=msg['content'])]
                    ))
        
        response = client.models.generate_content(
            model=model,
            contents=content_input,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=kwargs.get('max_tokens', 1024),
                temperature=kwargs.get('temperature', 0.7)
            )
        )
        
        return response.text
        
    except Exception as e:
        return f"Gemini API error: {str(e)}"

# Batch calling function
def call_llm_batch(
    model: str,
    queries: List[str] = None,
    messages_list: List[List[Dict[str, str]]] = None,
    system_prompt: str = "You are a helpful assistant.",
    **kwargs
) -> List[str]:
    """Batch call LLM"""
    results = []
    
    if queries:
        for query in queries:
            result = call_llm(model=model, query=query, system_prompt=system_prompt, **kwargs)
            results.append(result)
    elif messages_list:
        for messages in messages_list:
            result = call_llm(model=model, messages=messages, system_prompt=system_prompt, **kwargs)
            results.append(result)
    else:
        raise ValueError("Must provide either queries or messages_list parameter")
    
    return results

# Backward compatibility functions
def call_llms(*args, **kwargs):
    """Backward compatible function name"""
    return call_llm(*args, **kwargs)

def call_llms_batch(*args, **kwargs):
    """Backward compatible function name"""
    return call_llm_batch(*args, **kwargs)

# Usage examples
if __name__ == '__main__':
    # Test OpenAI/GPT model
    response1 = call_llm(
        model="gpt-4.1-nano-2025-04-14",
        query="Hello, please introduce yourself",
        system_prompt="You are a friendly AI assistant"
    )
    print("GPT Response:", response1)
    
    # Test Claude model
    response2 = call_llm(
        model="claude-sonnet-4-20250514",
        query="Hello, how are you?",
        system_prompt="You are a helpful assistant."
    )
    print("Claude Response:", response2)
    
    # Test Gemini model
    response3 = call_llm(
        model="gemini-2.5-flash",
        query="Who won Wimbledon this year?",
        system_prompt="You are a sports assistant."
    )
    print("Gemini Response:", response3)
    
    # Test DeepSeek model
    response4 = call_llm(
        model="deepseek-reasoner",
        query="Say hi",
        system_prompt="You are a helpful assistant."
    )
    print("DeepSeek Response:", response4)
    
    print("\n=== Batch Call Tests ===")
    
    # Batch test GPT model
    print("\n1. Testing GPT batch calls:")
    gpt_batch_results = call_llm_batch(
        model="gpt-4.1-nano-2025-04-14",
        queries=["Hello", "How are you?", "What's your name?"],
        system_prompt="You are a friendly chatbot."
    )
    print("GPT batch results:", gpt_batch_results)
    
    # Batch test Claude model
    print("\n2. Testing Claude batch calls:")
    claude_batch_results = call_llm_batch(
        model="claude-sonnet-4-20250514",
        queries=["Hello", "How are you?", "What's your name?"],
        system_prompt="You are a friendly chatbot."
    )
    print("Claude batch results:", claude_batch_results)
    
    # Batch test Gemini model
    print("\n3. Testing Gemini batch calls:")
    gemini_batch_results = call_llm_batch(
        model="gemini-2.5-flash",
        queries=["Hello", "How are you?", "What's your name?"],
        system_prompt="You are a friendly chatbot."
    )
    print("Gemini batch results:", gemini_batch_results)
    
    # Batch test DeepSeek model
    print("\n4. Testing DeepSeek batch calls:")
    deepseek_batch_results = call_llm_batch(
        model="deepseek-reasoner",
        queries=["Hello", "How are you?", "What's your name?"],
        system_prompt="You are a friendly chatbot."
    )
    print("DeepSeek batch results:", deepseek_batch_results)