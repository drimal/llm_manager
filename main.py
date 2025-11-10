from llm_manager.factory import LLMFactory
from llm_manager.exceptions import UnknownProviderError
from llm_manager.prompts.prompt_library import system_prompt
from llm_manager.reflection import ReflectiveLLMManager
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
    parser.add_argument("-m", "--model", type=str, default="nemotron-mini")
    parser.add_argument("-q", "--question", type=str, default="why is sky blue?")
    return parser.parse_args()




def main(args):
    provider_name = args.provider
    params = {"provider_name": provider_name}
    model = args.model
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
        model = "nemotron-mini"
    else:
        raise UnknownProviderError(f"Unsupported provider: {provider_name}")

    params["system_prompt"] = system_prompt
    client = LLMFactory.get_client(**params)
    reflecton_manager = ReflectiveLLMManager(llm_client=client)
    llm_config = {"model": model}

    response = reflecton_manager.reflect(
        user_query=query,
        reflection_strategy="self_critique",
        num_iterations=3
    )
    return response


if __name__ == "__main__":
    args = parse_arguments()
    response = main(args)
    print(response.model_dump_json(), indent=4)
