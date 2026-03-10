**app.py (Gradio Interface)**

Streaming Chat UI: Developed a web-based chat interface using Gradio that supports real-time token streaming.

Ollama API Integration: Connected the frontend to a local Ollama server (localhost:11434) using the requests library for backend 

inference.Context Handling: Implemented manual history processing to build a persistent conversational prompt for the model.

**Modelfile (Custom LLM Configuration)** 
Persona Definition: Created CodeGuru, an expert assistant persona focused on clear explanations and best practices.Fine-Tuning 

Parameters: Optimized model behavior with specific settings for temperature ($0.7$), top_p ($0.9$), and a $4096$ token context window.

Base Model: Leveraged CodeLlama as the foundation for specialized programming assistance.

![alt text](<Screenshot 2026-02-06 114516.png>)