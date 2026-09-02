from asyncio import tools
import math
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
load_dotenv(override=True)

model = init_chat_model("groq:openai/gpt-oss-20b")
import math

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.Use for addition operations"""
    return a + b

@tool
def multiply(a:float, b: float) -> float:
    """Multiply two numbers together.Use for multiplication operations"""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide one number by another.Use for division operations"""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract one number from another.Use for subtraction operations"""
    return a - b

@tool
def square_root(a: float)->str:
    """Calculate the square root of a number. Use this for square root operations"""
    return str(math.sqrt(a))    

tools=[add, multiply, divide, subtract, square_root]

print("____________Available Tools____________")
for t in tools:
    print(f"Tool Name: {t.name}, Description: {t.description}")
print()
agent = create_agent(
    model=model,
    tools=tools
)

def run_agent(question: str):
    """Run the agent and print the execution trace."""
    print(f"\n 🤨 User: {question}")
    print("-" * 60)
    result = agent.invoke({
        "messages": [("user",question)]
    })
    print("👓 Clean Agent Execution Trace")
    print("-" *60)

    step = 1

    for msg in result["messages"]:

        if msg.type == "human":
            print(f"{step}. User asked:")
            print(f"  {msg.content}")
            step += 1
        elif msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tool_call in msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"{step}. Agent decision: ")
                print(f"   I need to use the tool : {tool_name}")
                print(f"   Tool input: {tool_args}")
                step += 1
        elif msg.type == "tool":
            print(f"{step}. Tool observation:")
            print(f"   {msg.content}")
            step += 1

        elif msg.type == "ai" and msg.content:
            print(f"{step}. Agent final answer:")
            print(f"   {msg.content}")
            step += 1
    print("=" * 60 )

run_agent("What is 48+52?")
run_agent("What is 12*7?")
run_agent("What is 100 divided by 4?")
run_agent("What is 30 minus 18, then multiplied by 2?") 