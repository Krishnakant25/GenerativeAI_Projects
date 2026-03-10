**LLM Q&A Chatbot Suite**

This project features two interactive, Streamlit-based chatbot applications designed to provide users with a flexible interface for querying Large Language Models (LLMs). Whether you prefer using OpenAI's cloud-based models or running open-source models locally via Ollama, this suite has you covered.

**🚀 Features**
-- Dual-Engine Support: Choose between high-performance OpenAI models (GPT-4o, GPT-5-mini) or privacy-focused local models (Mistral).
-- LangSmith Integration: Built-in tracking and observability using LangChain’s LangSmith for monitoring and debugging.
-- Customizable Parameters: Real-time adjustment of Temperature and Max Tokens via the sidebar to control creativity and response length.
-- Modern UI: Clean and intuitive interface powered by Streamlit.

**🛠️ Project Structure** 

app.py: The main application script for the OpenAI-integrated chatbot.

ollama_chatbot.py: The application script for running local open-source models (e.g., Mistral) via Ollama.

requirements.txt: List of all necessary Python dependencies like LangChain and Streamlit.

.env: (User-created) Local environment file to store OPENAI_API_KEY and LANGCHAIN_API_KEY.

**📋 Prerequisites** 

Before running the applications, ensure you have:

1. Python 3.8+ installed.
2. An OpenAI API Key (for app.py).
3. Ollama installed and the mistral model pulled (for ollama_chatbot.py).
4. A LangChain API Key for tracing (optional but recommended).

**⚙️ Installation & Setup**

1. Clone the repository:

git clone <repository-url>

cd <repository-folder>

2. Install dependencies: 

pip install -r requirements.txt

3. Configure Environment Variables:

Create a .env file in the root directory and add your credentials:

LANGCHAIN_API_KEY=your_langchain_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

**🏃 Usage** 
1. OpenAI ChatbotTo run the version that uses OpenAI's cloud models: streamlit run app.py
Note: You can enter your OpenAI API key directly into the sidebar in the browser.

2. Ollama (Local) ChatbotTo run the local version:Bashstreamlit run ollama_chatbot.py
Note: Ensure the Ollama service is running on your machine.

**🧠 How It Works**

Both applications utilize LangChain Expression Language (LCEL) to chain the following components:

1. Prompt Template: Defines the system persona and user question format.

2. Model: Either ChatOpenAI or Ollama.

3. Output Parser: Converts the raw LLM output into a clean string.

chain = prompt | llm | output_parser