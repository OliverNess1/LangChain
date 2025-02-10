import os
import json
import ast
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults, Tool
from langchain.agents import initialize_agent, AgentType



os.environ["GROQ_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

# Initialize the AI model
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.0, max_retries=2)

# Set up the search tool (renaming it explicitly for AI)
search_tool = TavilySearchResults(max_results=3)
search_tool.name = "tavily_search"  # Explicitly renaming so the model calls it correctly

# Define a custom math tool (since LLMMathChain is deprecated)
def calculate(expression: str) -> str:
    """
    Safely evaluates a mathematical expression using Python's `ast.literal_eval`.
    Example input: "2.3 - 1.3"
    """
    try:
        result = ast.literal_eval(expression)
        return f"The result is {result}."
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Register the custom math tool
math_tool = Tool(
    name="calculate",
    func=calculate,
    description="Performs basic mathematical calculations. Input should be a numerical expression, e.g., '2.3 - 1.3'."
)

# Create an AI agent that can both look up data & perform calculations
agent = initialize_agent(
    tools=[search_tool, math_tool],  # Using our renamed search tool and math tool
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,  # Prevents crashes if AI misformats responses
)

# Interactive Loop for User Queries
print("\n🔹 AI Chatbot with Web Search & Math Enabled 🔹")
print("Type your question below. Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("Ask a question: ")

    # Exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Exiting AI Chat. Have a great day!")
        break

    # AI processes the query
    try:
        response = agent.run(user_input)
        print(f"\nAI Response: {response}\n")
    except Exception as e:
        print(f"\nError: {str(e)}\n")