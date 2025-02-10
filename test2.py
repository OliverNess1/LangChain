import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


load_dotenv("API.env")

# Initialize API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize the AI model
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, max_retries=2)

# Menu with Prices
menu = {
    "burger": 5.99,
    "pizza": 8.99,
    "soda": 1.99,
    "fries": 2.99,
    "salad": 4.49
}

# Shopping Cart (Initially Empty)
shopping_cart = {}

# Function: Add Item to Cart
def add_to_cart(args) -> str:
    """Adds an item to the shopping cart."""
    if isinstance(args, str):  
        try:
            args = json.loads(args.replace("'", '"'))  # Ensure valid JSON format
        except json.JSONDecodeError:
            return "Invalid input format. Please provide a valid item and quantity."

    item = args.get("item", "").lower()
    quantity = int(args.get("quantity", 1))  # Default to 1 if not provided

    if item not in menu:
        return f"{item} is not available on the menu."
    
    shopping_cart[item] = shopping_cart.get(item, 0) + quantity
    return f"Added {quantity}x {item}(s) to your cart."

# Function: Remove Item from Cart
def remove_from_cart(args) -> str:
    """Removes an item from the shopping cart."""
    if isinstance(args, str):  
        try:
            args = json.loads(args.replace("'", '"'))  # Ensure valid JSON format
        except json.JSONDecodeError:
            return "Invalid input format. Please provide a valid item and quantity."

    item = args.get("item", "").lower()
    quantity = int(args.get("quantity", 1))

    if item not in shopping_cart:
        return f"{item} is not in your cart."
    
    if shopping_cart[item] <= quantity:
        del shopping_cart[item]  
        return f"Removed all {item}(s) from your cart."
    else:
        shopping_cart[item] -= quantity
        return f"Removed {quantity}x {item}(s) from your cart."

# Function: View Cart
def view_cart(args=None) -> str:
    """Displays the current items in the shopping cart."""
    if not shopping_cart:
        return "Your shopping cart is empty."
    
    cart_summary = "\nShopping Cart:\n"
    total = 0
    for item, qty in shopping_cart.items():
        price = menu[item] * qty
        cart_summary += f" - {qty}x {item.capitalize()} (${price:.2f})\n"
        total += price
    
    cart_summary += f"\nTotal: ${total:.2f}"
    return cart_summary

# Register AI Tools
add_item_tool = Tool(
    name="add_to_cart",
    func=add_to_cart,
    description="Adds an item to the shopping cart. Requires 'item' (name) and 'quantity' (number, default is 1)."
)

remove_item_tool = Tool(
    name="remove_from_cart",
    func=remove_from_cart,
    description="Removes an item from the shopping cart. Requires 'item' (name) and 'quantity' (number)."
)

view_cart_tool = Tool(
    name="view_cart",
    func=view_cart,
    description="Displays the current shopping cart with items and total cost."
)

# Create an AI Agent
agent = initialize_agent(
    tools=[add_item_tool, remove_item_tool, view_cart_tool],  
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors="Please reformat your response.",  
)

message_history = [
    SystemMessage(content="You are a helpful AI assistant managing a shopping cart. Users may ask you to add or remove items, and check their cart. If you cannot find the exact item attempt to figure out what the user means, for example a hamburger is a burger, or two salads is two salad items."),
]

print("\nAI-Powered Shopping Cart")
print("You can ask the AI to add or remove items, and check your cart.")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("Your request: ")

    # Exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("\nThank you for shopping! Have a great day!")
        break

    # Store the user's message
    message_history.append(HumanMessage(content=user_input))

    # AI processes the query while keeping history
    try:
        response = agent.invoke({"input": message_history})
        ai_response = response["output"]

        # Store AI response in history
        message_history.append(AIMessage(content=ai_response))

        print(f"\nAI Response: {ai_response}\n")
    except Exception as e:
        print(f"\nError: {str(e)}\n")
