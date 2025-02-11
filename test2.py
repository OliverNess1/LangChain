import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from menu import menu 


load_dotenv("API.env")

# Initialize API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize the AI model
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, max_retries=2)

# Shopping Cart (Initially Empty)
shopping_cart = {}

def add_to_cart(args) -> str:
    """Adds an item to the shopping cart, allowing for modifications."""
    if isinstance(args, str):  
        try:
            args = json.loads(args.replace("'", '"'))
        except json.JSONDecodeError:
            return "Invalid input format. Please provide a valid item and quantity."

    item = args.get("item", "").strip().lower()
    quantity = int(args.get("quantity", 1))
    modifications = tuple(sorted(args.get("modifications", [])))  # Convert modifications to a tuple (sorted for consistency)

    if item not in menu:
        return f"Sorry, {item} is not available on the menu."

    # Generate a unique key based on item + modifications
    cart_key = (item, modifications)

    # Ensure correct structure
    if cart_key not in shopping_cart:
        shopping_cart[cart_key] = {"quantity": 0, "modifications": modifications}

    shopping_cart[cart_key]["quantity"] += quantity

    mod_text = f" with {' and '.join(modifications)}" if modifications else ""
    return f"Added {quantity}x {item}(s){mod_text} to your cart."


# Function: Remove Item from Cart
def remove_from_cart(args) -> str:
    """Removes a specific variant of an item from the shopping cart."""
    if isinstance(args, str):  
        try:
            args = json.loads(args.replace("'", '"'))
        except json.JSONDecodeError:
            return "Invalid input format. Please provide a valid item and quantity."

    item = args.get("item", "").strip().lower()
    quantity = int(args.get("quantity", 1))
    modifications = tuple(sorted(args.get("modifications", [])))

    cart_key = (item, modifications)

    if cart_key not in shopping_cart:
        return f"{item} with the specified modifications is not in your cart."

    if shopping_cart[cart_key]["quantity"] <= quantity:
        del shopping_cart[cart_key]  
        return f"Removed all {item}(s) {modifications} from your cart."
    else:
        shopping_cart[cart_key]["quantity"] -= quantity
        return f"Removed {quantity}x {item}(s) {modifications} from your cart."



def view_cart(args=None) -> str:
    """Displays the current items in the shopping cart."""
    if not shopping_cart:
        return "Your shopping cart is empty."

    cart_summary = "\nShopping Cart:\n"
    total = 0

    for (item, modifications), details in shopping_cart.items():
        qty = details.get("quantity", 1)
        mods = f" (Modified: {', '.join(modifications)})" if modifications else ""
        price = menu[item]["price"] * qty
        cart_summary += f" - {qty}x {item.capitalize()}{mods} (${price:.2f})\n"
        total += price

    cart_summary += f"\nTotal: ${total:.2f}"
    return cart_summary




def get_item_details(args) -> str:
    """Fetches the description and available modifications for a menu item."""

    if isinstance(args, str):  
        args = json.loads(args.replace("'", '"'))

    item = args.get("item", "").strip().lower()


    # Ensure item exists in the menu
    if item not in menu:
        print("DEBUG: Item not found in menu!")
        return f"Sorry, {item} is not available on the menu."

    details = menu[item]

    description = details.get("description", "No description available.")
    modifications = ", ".join(details.get("modifications", [])) or "None"

    return (
        f"Item: {item.capitalize()}\n"
        f"Description: {description}\n"
        f"Possible modifications: {modifications}\n"
    )


# Register AI Tools
get_item_details_tool = Tool(
    name="get_item_details",
    func=get_item_details,
    description="Retrieves the description and available modifications for a menu item. Requires the name of the item."
)

add_item_tool = Tool(
    name="add_to_cart",
    func=add_to_cart,
    description="Adds an item to the shopping cart. Requires 'item' (name), 'quantity' (number, default is 1), and optional 'modifications' (list of changes)."
)

remove_item_tool = Tool(
    name="remove_from_cart",
    func=remove_from_cart,
    description="Removes an item from the shopping cart. Requires 'item' (name) and 'quantity' (number)."
)

view_cart_tool = Tool(
    name="view_cart",
    func=view_cart,
    description="Displays the current shopping cart with items, modifications, and total cost."
)


# Create an AI Agent
agent = initialize_agent(
    tools=[get_item_details_tool, add_item_tool, remove_item_tool, view_cart_tool],  
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors="Please reformat your response.",  
)

message_history = [
    SystemMessage(content=(
        "You are a helpful AI assistant managing a shopping cart."
        " Users may ask you to add or remove items, and check their cart."
        " You can also provide item descriptions and possible modifications."
        " When answereing questions about the menu items, such as 'Does the burger have pickles?',"
        " always call the `get_item_details` function first."
        " If the user asks to perform a complex action, like adding two burgers with different modifications, add each item separately."
        " Then, use the description provided by `get_item_details()` to determine the correct response."
        " Do not make assumptions—only state what is explicitly mentioned in the description."
        " You must always structure your responses according to the following format:"
        " Thought: (Explain your reasoning)"
        " Action: (Choose one of the available functions)"
        " Action Input: (Provide input in valid JSON format)"
        " Observation: (Result from the function call)"
        " Final Answer: (Summarize the result in a natural response to the user)"
        " DO NOT return unstructured text. DO NOT use Markdown-style formatting (such as `<think>`)."
        " Always follow the structured response format. If you don't know what to do, retry instead of making up a response."
    ))
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
