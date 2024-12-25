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
