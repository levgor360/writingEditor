import streamlit as st
import replicate
import os
from anthropic import Anthropic  # Replace OpenAI import
from openai import OpenAI
import json
from phoenix.otel import register
import requests
from openinference.instrumentation.openai import OpenAIInstrumentor

# Phoenix API key
phoenix_api_key = "4402d414c66202f090f:9b282ce"
os.environ["PHOENIX_CLIENT_HEADERS"] = "api_key=4402d414c66202f090f:9b282ce"

# configure the Phoenix tracer
tracer_provider = register(
  project_name="my-llm-app", # Default is 'default'
  endpoint="https://app.phoenix.arize.com/v1/traces",
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# sidebar setup
with st.sidebar:
    # Title displayed on the side bar
    st.title('Parameters')
    # Request API key
    claude_api_key = st.text_input("API Key", key="chatbot_api_key", type="password")

client = Anthropic(api_key=claude_api_key)

chosen_temperature = st.sidebar.slider('temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.01)
chosen_max_tokens = st.sidebar.slider('max_tokens', min_value=32, max_value=4096, value=4096, step=8)

# main window title setup
st.subheader('Writing editor')

if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [{"role": "assistant", "content": "Decribe the task or problem you would like me to tackle."}]

# Show the items only after the system prompt on the front end
for message in st.session_state.messages[2:]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Make a button which clears the conversation and starts a new chat
def clear_chat_history():
    st.session_state["messages"] = [{"role": "assistant", "content": "Decribe the task or problem you would like me to tackle."}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def ClaudeAI_call(usr_prompt):
    
    st.chat_message("user").write(usr_prompt) # Display the user's message in the Streamlit chat interface.

    st.session_state.messages.append({"role": "user", "content": usr_prompt})

    with st.chat_message("assistant"):
      message_placeholder = st.empty()
      output = ""

      # Implement streaming functionality
      with client.messages.stream(
          model="claude-3-sonnet-20240229",
          messages=st.session_state.messages,
          temperature=chosen_temperature,
          max_tokens=chosen_max_tokens,
      ) as stream:
          for chunk in stream:
              if chunk.type == "content_block_delta":
                  output += chunk.delta.text
                  message_placeholder.write(output + " "
                  )
      
      message_placeholder.write(output)

    st.session_state.messages.append({"role": "assistant", "content": output})


if prompt := st.chat_input():
    if not claude_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    ClaudeAI_call(prompt)