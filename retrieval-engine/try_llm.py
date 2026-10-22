from src.llm_client import LLMClient


def main():
    client = LLMClient(
        model='google/gemma-4-31b-it',
        api_key_env='OPENROUTER_API_KEY',
    )
    test_msg = [{'role': 'user', 'content': 'What are the best python libraries for scalar product?'}]

    print(client.generate(test_msg))


if __name__ == '__main__':
    main()
