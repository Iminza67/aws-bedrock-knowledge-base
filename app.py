import streamlit as st
import boto3
from botocore.exceptions import ClientError
import json
from bedrock_utils import query_knowledge_base, generate_response, valid_prompt

# Streamlit UI
st.title("🧠 Bedrock Chat Application")

# Sidebar for configurations
st.sidebar.header("Configuration")
model_id = st.sidebar.selectbox(
    "Select LLM Model",
    [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0"
    ]
)
kb_id = st.sidebar.text_input("Knowledge Base ID", "your-knowledge-base-id")

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 0.9)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if valid_prompt(prompt,model_id):
        with st.spinner("Retrieving context from Knowledge Base..."):
            context = query_knowledge_base(kb_id, prompt)

        with st.spinner("Generating response..."):
            response = generate_response(prompt, context, model_id, temperature, top_p)
    else:
        response = "⚠️ I'm unable to process this prompt. Please try again."

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
