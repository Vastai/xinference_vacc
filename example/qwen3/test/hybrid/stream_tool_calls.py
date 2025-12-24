#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Stream function call test for vLLM/Xinference
"""

from openai import OpenAI
import argparse
import json
import time

def test_stream_function_call(messages, model_name, tools):
    """Test function calling in stream mode"""
    print(f"📝 User message: {messages[0]['content']}")
    print("\n📡 Starting stream response...")
    print("-" * 60)
    
    # Start streaming
    stream_start = time.time()
    stream_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        stream=True
    )
    
    # Variables to collect stream data
    collected_content = ""
    collected_tool_calls = {}
    finish_reason = None
    
    # Process stream chunks
    chunk_count = 0
    for chunk in stream_response:
        chunk_count += 1
        
        if chunk.choices:
            choice = chunk.choices[0]
            delta = choice.delta
            
            # Collect content
            if delta.content:
                collected_content += delta.content
                print(f"💬 Content chunk: {delta.content}", end="", flush=True)
            
            # Collect tool calls
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    idx = tool_call.index
                    
                    # Initialize if not exists
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {
                            "id": "",
                            "type": "",
                            "function": {"name": "", "arguments": ""}
                        }
                    
                    # Update tool call data
                    if tool_call.id:
                        collected_tool_calls[idx]["id"] = tool_call.id
                        print(f"\n🆔 Tool call ID: {tool_call.id}")
                    
                    if tool_call.type:
                        collected_tool_calls[idx]["type"] = tool_call.type
                    
                    if tool_call.function:
                        if tool_call.function.name:
                            collected_tool_calls[idx]["function"]["name"] += tool_call.function.name
                            print(f"\n🛠️  Function name: {tool_call.function.name}")
                        
                        if tool_call.function.arguments:
                            collected_tool_calls[idx]["function"]["arguments"] += tool_call.function.arguments
                            print(f"📄 Arguments chunk: {tool_call.function.arguments}", end="", flush=True)
            
            # Check finish reason
            if choice.finish_reason:
                finish_reason = choice.finish_reason
    
    stream_end = time.time()
    print(f"\n" + "-" * 60)
    print(f"✅ Stream completed in {stream_end - stream_start:.2f} seconds")
    print(f"📊 Total chunks received: {chunk_count}")
    
    # Display collected content
    if collected_content:
        print(f"\n📝 Complete content: {collected_content}")
    # Display collected tool calls
    if collected_tool_calls:
        print(f"\n🔧 Detected {len(collected_tool_calls)} tool call(s):")
        
        for idx, tool_data in sorted(collected_tool_calls.items()):
            print(f"\n  Tool #{idx + 1}:")
            print(f"    ID: {tool_data['id']}")
            print(f"    Type: {tool_data['type']}")
            print(f"    Function: {tool_data['function']['name']}")
            
            if tool_data['function']['arguments']:
                print(f"    Arguments: {tool_data['function']['arguments']}")
                
                # Try to parse JSON
                try:
                    parsed_args = json.loads(tool_data['function']['arguments'])
                    print(f"    Parsed JSON: {json.dumps(parsed_args, indent=4, ensure_ascii=False)}")
                except json.JSONDecodeError:
                    print(f"    ⚠️ Arguments are not valid JSON (may be incomplete)")
                    print(f"    Raw arguments: {tool_data['function']['arguments']}")
            else:
                print(f"    ⚠️ No arguments provided")
        
        return collected_tool_calls, collected_content
    else:
        print(f"\n❌ No tool calls detected in stream response")
        return None, collected_content

def test_complete_conversation_stream(messages, model_name, tools):
    """Test complete conversation with tool execution in stream mode"""
    print("\n" + "=" * 80)
    print("🔄 Testing COMPLETE CONVERSATION (Stream Mode)")
    print("=" * 80)
    
    # Step 1: Get tool call from model
    print("\n📤 Step 1: Getting tool call from model...")
    tool_calls, content = test_stream_function_call(messages, model_name, tools)
    
    if not tool_calls:
        print("❌ Cannot continue conversation: No tool calls received")
        return
    
    # Step 2: Add assistant message to history
    print("\n📝 Step 2: Adding assistant response to conversation...")
    
    # Get first tool call
    first_tool_idx = sorted(tool_calls.keys())[0]
    first_tool = tool_calls[first_tool_idx]
    
    # Construct assistant message
    assistant_message = {
        "role": "assistant",
        "content": content,
        "tool_calls": []
    }
    
    # Add all tool calls
    for idx, tool_data in sorted(tool_calls.items()):
        assistant_message["tool_calls"].append({
            "id": tool_data["id"] or f"call_{idx}",
            "type": tool_data["type"] or "function",
            "function": {
                "name": tool_data["function"]["name"],
                "arguments": tool_data["function"]["arguments"]
            }
        })
    
    messages.append(assistant_message)
    print(f"✅ Added assistant message with {len(tool_calls)} tool call(s)")
    
    # Step 3: Simulate tool execution
    print("\n⚡ Step 3: Simulating tool execution...")
    
    # Try to parse location from arguments
    location = "Unknown"
    try:
        args = json.loads(first_tool["function"]["arguments"])
        location = args.get("location", "Unknown")
    except:
        # Try to extract location from raw arguments string
        import re
        location_match = re.search(r'"location"\s*:\s*"([^"]+)"', first_tool["function"]["arguments"])
        if location_match:
            location = location_match.group(1)
    
    # Mock weather data
    weather_data = {
        "location": location,
        "temperature": "24℃",
        "condition": "Sunny",
        "humidity": "65%",
        "wind_speed": "12 km/h",
        "forecast": "Clear skies throughout the day"
    }
    
    tool_result = json.dumps(weather_data, ensure_ascii=False)
    
    # Add tool result to messages
    messages.append({
        "role": "tool",
        "tool_call_id": first_tool["id"] or "call_0",
        "content": tool_result
    })
    
    print(f"📍 Location: {location}")
    print(f"📊 Tool result: {tool_result}")
    
    # Step 4: Get final response
    print("\n💭 Step 4: Getting model's final response...")
    
    final_stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True
    )
    
    final_content = ""
    for chunk in final_stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content_chunk = chunk.choices[0].delta.content
            print(content_chunk, end="", flush=True)
            final_content += content_chunk
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream function call test")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port number (default: 8000)")
    parser.add_argument("--model-name", type=str, default="qwen2.5-7b",
                    help="Model name (default: qwen2.5-7b)")
    
    args = parser.parse_args()

    # Initialize client
    client = OpenAI(
        base_url="http://" + args.host + ":" + str(args.port) + "/v1",
        api_key="token-abc123"
    )

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather of a location, the user should supply a location first",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit (optional)"
                        }
                    },
                    "required": ["location"]
                },
            }
        },
    ]

    # Base message for testing
    base_messages = [{"role": "user", "content": "How's the weather in Beijing?"}]
        
    test_complete_conversation_stream(base_messages.copy(), args.model_name, tools)
    