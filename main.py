from llm_manager.factory import LLMFactory
import os
import sys
from dotenv import load_dotenv

load_dotenv()


provider_name = sys.argv[1]
params = {"provider_name": provider_name}
if provider_name == "openai":
    params["api_key"] = os.getenv("OPENAI_API_KEY")
    model = "gpt-4o-mini"
elif provider_name == "anthropic":
    params["api_key"] = os.getenv("ANTHROPIC_API_KEY")
elif provider_name == "bedrock":
    params["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
    params["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    params["region_name"] = os.getenv("AWS_REGION")
    model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
elif provider_name == "ollama":
    params["base_url"] = os.getenv("OLLAMA_BASE_URL")
    model = "nemotron-mini"


print(params)
client = LLMFactory.get_client(**params)
prompt = "why is sky blue?"
response = client.generate(prompt=prompt)
print(response)
