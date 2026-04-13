import streamlit as st
#from langchain_openai.chat_models import ChatOpenAI
#from dataclasses import dataclass
#from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
#from langchain.tools import tool, ToolRuntime
#from langgraph.checkpoint.memory import InMemorySaver
#from langchain.agents.structured_output import ToolStrategy
#from langsmith.wrappers import wrap_openai
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse

# Initialize Langfuse client
langfuse = Langfuse()

st.title("🦜🔗 Local Bot")

#openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
# Initialize lanfuse callback handler for monitoring and debugging
langfuse_handler = CallbackHandler()

# Fetch default production version prompt from Langfuse
langfuse_prompt = langfuse.get_prompt("dev/persona", type="chat")

# Configure model
model = init_chat_model(
    model="mistralai/ministral-3-3b",
    model_provider='openai', # because LM Studio mimics OpenAI's API
    base_url='http://127.0.0.1:1234/v1',
    api_key='not-needed', # LM Studio accepts any string here
    # system_prompt = "You are a helpful assistant who answers like a pirate."
)

# create a prompt template for the bot and give it a system prompt
## Instead of defining a system prompt ourselves, we can use the one from Langfuse which is already being used to monitor the bot's performance. 
# This way, we ensure that the bot's behavior is consistent with what we're monitoring in Langfuse.
# This makes Prompt Management easier as we only have to manage one prompt in Langfuse, and it also allows us to leverage any additional features
#  or context that the Langfuse prompt might include for better monitoring and debugging. 
# #prompt_template = ChatPromptTemplate.from_messages([
   # ("system", langfuse_prompt.get_langchain_prompt()),
   # ("user", "{input}")
## ])
prompt_template = ChatPromptTemplate.from_messages(langfuse_prompt.get_langchain_prompt())

# Chain the prompt with the model initialized via init_chat_model
chain = prompt_template | model

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Iterate through chat history messages and render their contents in chat
# message containers
for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
def generate_response(input_text: str) -> str:
    #model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
    #response = model.invoke(input_text)
    response = chain.invoke(input_text,config={"callbacks": [langfuse_handler]})
    return response.content

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        st.write("thinking...")
        response = st.write(generate_response(str(prompt)))
    st.session_state.history.append({"role": "assistant", "content": response})
