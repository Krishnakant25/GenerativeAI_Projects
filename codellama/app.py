import requests
import json
import gradio as gr

url = "http://localhost:11434/api/generate"

def generate_response(message, history):
    """
    Generate response from Ollama API.
    
    Args:
        message: Current user message (str)
        history: Chat history as list of [user_msg, bot_msg] pairs
    
    Yields:
        Partial message as it streams from the API
    """
    # Build prompt from history
    full_prompt = ""
    
    # Process history
    for user_msg, assistant_msg in history:
        full_prompt += f"User: {user_msg}\n"
        full_prompt += f"Assistant: {assistant_msg}\n"
    
    # Add current message
    full_prompt += f"User: {message}\nAssistant: "
    
    data = {
        "model": "codeguru",
        "prompt": full_prompt,
        "stream": True
    }
    
    try:
        response = requests.post(url, json=data, stream=True, timeout=60)
        response.raise_for_status()
        
        partial_message = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    partial_message += token
                    yield partial_message
                except json.JSONDecodeError:
                    continue
                    
    except requests.exceptions.Timeout:
        yield "Error: Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        yield "Error: Cannot connect to Ollama server. Is it running on localhost:11434?"
    except requests.exceptions.RequestException as e:
        yield f"Error connecting to Ollama: {str(e)}"

# Create ChatInterface for Gradio 6.x
demo = gr.ChatInterface(
    fn=generate_response,
    title="CodeGuru Assistant",
    description="Ask me anything about coding! Powered by Ollama.",
    examples=[
        "How do I create a Python class?",
        "Explain async/await in JavaScript",
        "What's the difference between SQL joins?"
    ],
    fill_height=True
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
