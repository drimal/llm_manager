from llm_manager.factory import LLMFactory
from llm_manager.exceptions import UnknownProviderError
from llm_manager.prompts.prompt_library import system_prompt
from llm_manager.reflection import ReflectiveLLMManager
from llm_manager.providers.provider_registry import ProviderRegistry
from pathlib import Path
import os
import sys
from dotenv import load_dotenv
from argparse import ArgumentParser
import logging
import json

load_dotenv()
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-p", "--provider", type=str, required=False, default="ollama")
    #parser.add_argument("-m", "--model", type=str, default="nemotron-mini")
    parser.add_argument("-q", "--question", type=str, default="why is sky blue?")
    return parser.parse_args()

def main(args):
    provider_name = args.provider
    params = {"provider_name": provider_name}
    model = None #args.model
    query = args.question

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
        #model = "nemotron-mini"
        model = "gpt-oss:120b-cloud"
    else:
        raise UnknownProviderError(f"Unsupported provider: {provider_name}")

    params["system_prompt"] = system_prompt
    config_path = Path.path("./config/config.yaml")
    provider_registry = ProviderRegistry(config_path)

    # Run eval on specific model
    model_name = provider_registry.configure_for_model("gpt4o-mini", params)

    print(model_name)
    
    llm_client = LLMFactory.get_client(**params)

    reflection_manager = ReflectiveLLMManager(llm_client=llm_client)
    #llm_config = {"model": model}
    
    response = reflection_manager.reflect(
        user_query=query,
        reflection_strategy="self_critique",
        num_iterations=3
    )
    
    #response = llm_client.generate(prompt=query, model=model)
    return response

if __name__ == "__main__":
    args = parse_arguments()
    response = main(args)
    response = json.loads(response.model_dump_json())
    print(response.get("text"))
    print(response.get("usager"))

    sys.exit(0)
