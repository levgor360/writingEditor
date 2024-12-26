import streamlit as st
import replicate
import os
from anthropic import Anthropic  # Replace OpenAI import
from openai import OpenAI
import json
import yaml
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

# Load YAML file
yaml_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
with open(yaml_path, 'r') as file:
    prompts = yaml.safe_load(file)

## Testing:
# st.write(prompts['system_prompt'])

# sidebar setup
with st.sidebar:
    # Title displayed on the side bar
    st.title('Parameters')
    # Request API key
    claude_api_key = st.text_input("API Key", key="chatbot_api_key", type="password")
    # Sentence correction checkbox
    enable_sentence_correction = st.sidebar.checkbox('Enable Sentence Polisher', value=False)

client = Anthropic(api_key=claude_api_key)

chosen_temperature = st.sidebar.slider('temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.01)

# main window title setup
st.subheader('Writing editor')

if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Decribe the task or problem you would like me to tackle."}
        ]

# Show the items only after the system prompt on the front end
for message in st.session_state.messages[2:]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Make a button which clears the conversation and starts a new chat
def clear_chat_history():
    st.session_state["messages"] = [{"role": "assistant", "content": "Decribe the task or problem you would like me to tackle."}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Define the call
def ClaudeAI_call(usr_prompt):
    # Add debug logging
    system_prompt = prompts['system_prompt']
    st.sidebar.text(f"System prompt approximate tokens: {len(system_prompt.split())}")
    
    # Log current conversation history size
    history_text = " ".join([m["content"] for m in st.session_state.messages])
    st.sidebar.text(f"History approximate tokens: {len(history_text.split())}")
    
    st.chat_message("user").write(usr_prompt) # Display the user's message in the Streamlit chat interface.
    try:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            output = ""

            # Implement streaming functionality
            with client.messages.stream(
                model="claude-3-sonnet-20240229",
                messages=st.session_state.messages,
                system=prompts['system_prompt'],
                temperature=chosen_temperature,
                max_tokens=4096,
            ) as stream:
                for chunk in stream:
                    if chunk.type == "content_block_delta":
                        output += chunk.delta.text
                        message_placeholder.write(output + " "
                        )
        # Calculate and display total tokens after receiving the complete response
        system_prompt = prompts['system_prompt']
        history_text = " ".join([m["content"] for m in st.session_state.messages])
        response_tokens = len(output.split())
        total_tokens = len(system_prompt.split()) + len(history_text.split()) + response_tokens
        st.sidebar.text(f"Total tokens: {total_tokens}")
    except Exception as e:
        st.error(f"API Error: {str(e)}")
      
    message_placeholder.write(output)
    st.session_state.messages.append({"role": "assistant", "content": output})

def editor_chain(usr_prompt):
    if len(st.session_state.messages) == 1 and enable_sentence_correction == True: #apply the backend prompt to the users input, but only on their first input
        sentence_mod_prompt = prompts['sentence_correction'].replace("{user_input}", usr_prompt)
        st.session_state.messages.append({"role": "user", "content": sentence_mod_prompt})
    else:
        st.session_state.messages.append({"role": "user", "content": str((usr_prompt))})    
    
    ClaudeAI_call(usr_prompt)

if prompt := st.chat_input():
    if not claude_api_key:
        st.info("Please add your API key to continue.")
        st.stop()
    editor_chain(prompt)