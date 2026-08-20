import os
import json
import requests
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment configuration
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment.")

# Initialize the Gemini GenAI SDK client
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-3.5-flash"

app = Flask(__name__)
# Secure session signing key
app.secret_key = os.urandom(24)

# ============================================================
# TOOL FUNCTIONS
# ============================================================

def add_numbers(a: float, b: float) -> dict:
    """
    Add two numbers together. Used for calculations.
    """
    return {
        "operation": "addition",
        "a": a,
        "b": b,
        "result": a + b
    }


def product_info(product_name: str) -> dict:
    """
    Get specifications and pricing details for a product (e.g. iPhone 15, MacBook Air).
    """
    products = {
        "iphone 15": {
            "name": "iPhone 15",
            "category": "Smartphone",
            "price": "₹69,999",
            "rating": "4.6/5",
            "status": "In Stock"
        },
        "samsung s24": {
            "name": "Samsung Galaxy S24",
            "category": "Smartphone",
            "price": "₹74,999",
            "rating": "4.7/5",
            "status": "In Stock"
        },
        "macbook air": {
            "name": "MacBook Air (M3)",
            "category": "Laptop",
            "price": "₹99,999",
            "rating": "4.8/5",
            "status": "In Stock"
        }
    }
    
    key = product_name.lower().strip()
    product = products.get(key)
    if product:
        return product
    return {"error": f"Product '{product_name}' not found."}


def get_weather(city: str) -> dict:
    """
    Get live weather data including temperature, conditions, and wind speed for a city.
    """
    if not WEATHER_API_KEY:
        return {"error": "Weather service API key is not configured."}
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "city": data.get("name"),
                "temperature": f"{data.get('main', {}).get('temp')}°C",
                "condition": data.get("weather", [{}])[0].get("description", "Unknown").title(),
                "humidity": f"{data.get('main', {}).get('humidity')}%",
                "wind_speed": f"{data.get('wind', {}).get('speed')} m/s"
            }
        else:
            return {"error": f"City '{city}' not found or service error."}
    except Exception as e:
        return {"error": f"Failed to retrieve weather data: {str(e)}"}

# ============================================================
# FUNCTION REGISTRY
# ============================================================

AVAILABLE_FUNCTIONS = {
    "add_numbers": add_numbers,
    "product_info": product_info,
    "get_weather": get_weather
}

# ============================================================
# GEMINI TOOL DECLARATIONS
# ============================================================

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="add_numbers",
        description="Add two numbers together. Best for math queries, equations, and additions.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "a": types.Schema(type="NUMBER", description="First input number."),
                "b": types.Schema(type="NUMBER", description="Second input number.")
            },
            required=["a", "b"]
        )
    ),
    types.FunctionDeclaration(
        name="product_info",
        description="Query the product catalog for pricing, specs, and stock details of matching phones or laptops.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "product_name": types.Schema(type="STRING", description="Model or generic name of the product.")
            },
            required=["product_name"]
        )
    ),
    types.FunctionDeclaration(
        name="get_weather",
        description="Fetch live temperature and weather updates for a specified city name.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "city": types.Schema(type="STRING", description="Target city name.")
            },
            required=["city"]
        )
    )
]

CONFIG = types.GenerateContentConfig(
    system_instruction='''
# Gemini 2.5/3.6 Flash — Advanced Tool-Calling Agent Master Prompt

You are an advanced AI agent powered by **Gemini** with access to a dynamic collection of Python tools/functions.

Your primary responsibility is to understand the user's natural-language request, determine whether one or more tools are required, select the most appropriate tool(s), provide valid arguments, execute the requested operations through the available tool system, and then return a clear, accurate, user-friendly final response.

## 1. CORE OBJECTIVE
Act as an intelligent tool-using AI assistant.
For every user request:
- Determine whether a tool is necessary.
- Call the tool. Do not fabricate results.
- Incorporate results into final user-friendly responses.

## 2. TOOLS
- `add_numbers(a, b)`: For calculations
- `product_info(product_name)`: For smartphone or laptop specs
- `get_weather(city)`: For live weather updates in cities
''',
    tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)]
)

# ============================================================
# STATEFUL IN-MEMORY CONVERSATION DATABASE
# ============================================================

CONVERSATIONS = {}

# ============================================================
# ENDPOINTS
# ============================================================

@app.route("/")
def index():
    import uuid
    # Initialize a new stateful session ID and clean conversation
    chat_id = str(uuid.uuid4())
    session["chat_id"] = chat_id
    CONVERSATIONS[chat_id] = []
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"reply": "Empty message received"}), 400
    
    chat_id = session.get("chat_id")
    if not chat_id or chat_id not in CONVERSATIONS:
        import uuid
        chat_id = str(uuid.uuid4())
        session["chat_id"] = chat_id
        CONVERSATIONS[chat_id] = []
        
    history = CONVERSATIONS[chat_id]
    
    # Append the user's new message to the stateful history
    history.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        )
    )
    
    executed_logs = []
    
    try:
        while True:
            # Generate the content response
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=history,
                config=CONFIG
            )
            
            candidate = response.candidates[0]
            model_content = candidate.content
            
            tool_calls_found = False
            for part in model_content.parts:
                if part.function_call:
                    tool_calls_found = True
                    
            if tool_calls_found:
                # Appending the exact candidate.content preserves the thought signatures
                history.append(model_content)
                
                # Execute each function call sequentially
                for part in model_content.parts:
                    if part.function_call:
                        fn = part.function_call
                        fn_name = fn.name
                        fn_args = dict(fn.args)
                        
                        executed_logs.append({
                            "status": "calling",
                            "tool": fn_name,
                            "message": f"Calling tool `{fn_name}` with parameters: {json.dumps(fn_args)}"
                        })
                        
                        # run
                        if fn_name in AVAILABLE_FUNCTIONS:
                            result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)
                        else:
                            result = {"error": f"Unknown function: {fn_name}"}
                            
                        executed_logs.append({
                            "status": "success",
                            "tool": fn_name,
                            "message": f"Tool `{fn_name}` output: {json.dumps(result)}"
                        })
                        
                        # Feed the function response back under 'user' role
                        history.append(
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_function_response(
                                        name=fn_name,
                                        response=result
                                    )
                                ]
                            )
                        )
                # Continue generating
                continue
            
            else:
                # Text response is ready
                history.append(model_content)
                
                return jsonify({
                    "reply": response.text or "Execution completed.",
                    "logs": executed_logs
                })
                
    except Exception as e:
        import traceback
        with open("error_trace.txt", "w") as f:
            f.write(traceback.format_exc())
        return jsonify({
            "reply": f"An error occurred: {str(e)}",
            "logs": executed_logs
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

