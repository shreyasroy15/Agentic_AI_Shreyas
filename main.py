import os
import json
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment configuration
load_dotenv()

# App configuration
DEBUG_MODE = os.getenv("FLASK_DEBUG", "false").lower() == "true"
DEBUG_TOOLS = os.getenv("DEBUG_TOOLS", "false").lower() == "true"
PORT = int(os.getenv("PORT", 10000))

# Configure basic console logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
# Enable CORS for frontend cross-origin requests
CORS(app, resources={r"/*": {"origins": "*"}})

# Validate Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY is not set in environment.")
    raise ValueError("GEMINI_API_KEY environment variable is required.")

# Model configuration: default fallback is gemini-3.5-flash for quota reasons
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# INITIAL TOOLS
# ============================================================

def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first number."""
    return a - b

def product(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide the first number by the second number."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

def power(a: float, b: float) -> float:
    """Calculate the first number raised to the power of the second number."""
    return a ** b

def get_weather(city: str) -> dict:
    """
    Get weather information for a city.
    Note: Connect to a real weather service API in production.
    """
    fake_weather_data = {
        "delhi": {"temperature": "32°C", "condition": "Sunny", "humidity": "45%"},
        "mumbai": {"temperature": "29°C", "condition": "Cloudy", "humidity": "72%"},
        "bangalore": {"temperature": "24°C", "condition": "Partly Cloudy", "humidity": "68%"},
        "kolkata": {"temperature": "30°C", "condition": "Humid", "humidity": "78%"},
        "london": {"temperature": "17°C", "condition": "Drizzle", "humidity": "85%"},
        "new york": {"temperature": "22°C", "condition": "Sunny", "humidity": "50%"}
    }
    
    key = city.lower().strip()
    return fake_weather_data.get(key, {
        "temperature": "Unknown",
        "condition": "Weather data unavailable for this city",
        "humidity": "Unknown"
    })

# ============================================================
# TOOL REGISTRY AND GEMINI DECLARATIONS
# ============================================================

AVAILABLE_FUNCTIONS = {
    "add": add,
    "subtract": subtract,
    "product": product,
    "divide": divide,
    "power": power,
    "get_weather": get_weather
}

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="add",
        description="Add two numbers together (performs a + b). Used for arithmetic operations.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "a": types.Schema(type="NUMBER", description="First number"),
                "b": types.Schema(type="NUMBER", description="Second number")
            },
            required=["a", "b"]
        )
    ),
    types.FunctionDeclaration(
        name="subtract",
        description="Subtract the second number (b) from the first number (a). Used for equations or diffs.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "a": types.Schema(type="NUMBER", description="First number"),
                "b": types.Schema(type="NUMBER", description="Number to subtract")
            },
            required=["a", "b"]
        )
    ),
    types.FunctionDeclaration(
        name="product",
        description="Multiply two numbers together (performs a * b). Used for multiplication queries.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "a": types.Schema(type="NUMBER", description="First number"),
                "b": types.Schema(type="NUMBER", description="Second number")
            },
            required=["a", "b"]
        )
    ),
    types.FunctionDeclaration(
        name="divide",
        description="Divide the dividend (a) by the divisor (b). Throws ValueError on division by zero.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "a": types.Schema(type="NUMBER", description="Dividend number"),
                "b": types.Schema(type="NUMBER", description="Divisor number")
            },
            required=["a", "b"]
        )
    ),
    types.FunctionDeclaration(
        name="power",
        description="Calculate base (a) raised to exponent power (b) (performs a ** b). Used for power calculations.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "a": types.Schema(type="NUMBER", description="Base number"),
                "b": types.Schema(type="NUMBER", description="Exponent number")
            },
            required=["a", "b"]
        )
    ),
    types.FunctionDeclaration(
        name="get_weather",
        description="Retrieve weather information (temp, conditions, etc.) for a specified city name.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "city": types.Schema(type="STRING", description="Name of the target city")
            },
            required=["city"]
        )
    )
]

CONFIG = types.GenerateContentConfig(
    system_instruction="""
You are an intelligent AI assistant with access to Python tools.

Your responsibilities are:
1. Understand the user's intent.
2. Determine whether a tool is necessary.
3. Select the most appropriate available tool.
4. Extract valid arguments.
5. Call the appropriate tool.
6. Never fabricate tool results.
7. Use actual tool results in the final response.
8. If multiple tools are required, execute them in the correct order.
9. If no tool is required, answer normally.
10. Never expose internal reasoning or hidden chain-of-thought.

Be accurate, concise, and helpful.
""",
    tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)]
)

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/", methods=["GET"])
def home():
    """Return API basic status."""
    return jsonify({
        "status": "online",
        "message": "Gemini Tool Agent is running"
    }), 200

@app.route("/health", methods=["GET"])
def health():
    """Return health status check."""
    return jsonify({
        "status": "healthy"
    }), 200

@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle natural-language requests, execute matching tools,
    and return natural-language answers in JSON.
    """
    # 1. Validation Checks
    if not request.is_json:
        return jsonify({
            "success": false,
            "error": "Request body must be a valid JSON object."
        }), 400

    data = request.get_json()
    message = data.get("message")
    
    if message is None:
        return jsonify({
            "success": false,
            "error": "Missing 'message' field in JSON request."
        }), 400
        
    if not isinstance(message, str):
        return jsonify({
            "success": false,
            "error": "'message' field must be a string."
        }), 400
        
    if not message.strip():
        return jsonify({
            "success": false,
            "error": "'message' field cannot be blank."
        }), 400

    # 2. Conversation loop
    history = [
        types.Content(
            role="user",
            parts=[types.Part(text=message)]
        )
    ]
    
    try:
        max_turns = 10  # Prevent accidental infinite backend loops
        turns = 0
        
        while turns < max_turns:
            turns += 1
            
            # API call to Gemini model
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=history,
                config=CONFIG
            )
            
            # Extract content candidates
            if not response.candidates or not response.candidates[0].content:
                raise RuntimeError("Empty response received from the Gemini API model.")
                
            candidate = response.candidates[0]
            model_content = candidate.content
            
            # Check for tool call requests
            tool_calls_requested = False
            for part in model_content.parts:
                if part.function_call:
                    tool_calls_requested = True
                    
            if tool_calls_requested:
                # Append exact candidate content to preserve thought signature
                history.append(model_content)
                
                # Execute each function call sequentially
                for part in model_content.parts:
                    if part.function_call:
                        fn = part.function_call
                        fn_name = fn.name
                        fn_args = dict(fn.args) if fn.args else {}
                        
                        # Validate if tool exists in registry
                        if fn_name not in AVAILABLE_FUNCTIONS:
                            logger.error(f"Security Alert: Model requested unregistered tool '{fn_name}'")
                            raise ValueError(f"Unregistered function execution blocked: {fn_name}")
                            
                        # Optional Debug Logging
                        if DEBUG_TOOLS:
                            logger.info(f"TOOL CALL | Tool: {fn_name} | Arguments: {json.dumps(fn_args)}")
                            
                        # Execute registered python function
                        try:
                            result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)
                        except Exception as tool_err:
                            logger.error(f"Error executing tool '{fn_name}': {str(tool_err)}")
                            result = {"error": f"Tool execution failed: {str(tool_err)}"}
                            
                        if DEBUG_TOOLS:
                            logger.info(f"TOOL RESULT | Tool: {fn_name} | Result: {json.dumps(result)}")

                        # Feed the function response back under 'user' role
                        history.append(
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_function_response(
                                        name=fn_name,
                                        response={"result": result}
                                    )
                                ]
                            )
                        )
                # Continue generator execution
                continue
                
            else:
                # Generation complete: final natural-language response ready
                reply_text = response.text or "Execution completed."
                return jsonify({
                    "success": True,
                    "response": reply_text
                }), 200
                
        raise RuntimeError("Maximum reasoning loop turns reached without model completion.")

    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        logger.error(f"Error in /chat routine: {err_trace}")
        
        # Gracefully handle 429 API quota limits without showing internal stack traces
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "RESOURCE_EXHAUSTED" in err_trace:
            return jsonify({
                "success": False,
                "error": "Gemini API quota has been exhausted (429 RESOURCE_EXHAUSTED). Please wait a moment and try again."
            }), 429
            
        # Do not expose internal stack traces or API keys to the response
        return jsonify({
            "success": False,
            "error": "Failed to complete request due to an internal server error."
        }), 500

if __name__ == "__main__":
    logger.info(f"Starting Gemini Tool Agent on 0.0.0.0:{PORT} in debug={DEBUG_MODE}")
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
