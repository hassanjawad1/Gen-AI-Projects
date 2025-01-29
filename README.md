# Gen-AI-Projects

---

# AI Chatbot Agents  

**AI Chatbot Agents** is a Streamlit-based application that allows users to create and interact with AI-powered agents. It supports multiple AI models from different providers, including Groq and OpenAI, with optional web search capabilities.  

## 🚀 Features  
- Select AI model provider (Groq or OpenAI)  
- Choose from available models (e.g., Mixtral-8x7b-32768, GPT-4o-mini)  
- Define a system prompt to customize agent behavior  
- Enable or disable web search for real-time information retrieval  
- User-friendly Streamlit UI for seamless interaction  

## 📦 Installation  

### Prerequisites  
- Python 3.8+  
- Streamlit  
- `requests` library  

### Setup  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/ai-chatbot-agents.git
   cd ai-chatbot-agents
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```  

## ⚙️ Usage  
1. Open the app in a browser.  
2. Define your AI agent by entering a system prompt.  
3. Select a model provider (Groq/OpenAI).  
4. Choose an AI model from the dropdown.  
5. (Optional) Enable web search for real-time responses.  
6. Enter your query and click "Ask Agent!"  


```  
The payload includes:  
- `model_name`: Selected AI model  
- `model_provider`: Chosen provider (Groq/OpenAI)  
- `system_prompt`: User-defined system prompt  
- `messages`: User query  
- `allow_search`: Boolean for web search  


## 📞 Contact  
For any questions or contributions, feel free to reach out!  

