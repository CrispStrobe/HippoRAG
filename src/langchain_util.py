import os

import tiktoken

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


def num_tokens_by_tiktoken(text: str):
    return len(enc.encode(text))


    """
    Initialize a language model from the langchain library.
    :param llm: The LLM to use, e.g., 'openai', 'together'
    :param model_name: The model name to use, e.g., 'gpt-3.5-turbo'
    """
    if llm == 'openai':
        try:
            # Get the API key and model name as exported environment parameters
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            
            # verbose check the creation parameters
            print(f"Creating ChatOpenAI with: model: {model_name}, API_KEY: {api_key}, BASE_URL: {base_url}.\n")
            
            # Initialize the ChatOpenAI instance
            model = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name, temperature=temperature, max_retries=max_retries, timeout=timeout, **kwargs)
            
            # Check if the model object is created correctly
            if model and hasattr(model, 'model'):
                print("ChatOpenAI model created successfully.\n")
                return model
            else:
                raise ValueError("Failed to create ChatOpenAI model.")    
        except Exception as e:
            print(f"Error initializing ChatOpenAI: {e}\n")
            return None
    elif llm == 'together':
        # https://python.langchain.com/v0.1/docs/integrations/chat/together/
        from langchain_together import ChatTogether
        return ChatTogether(api_key=os.environ.get("TOGETHER_API_KEY"), model=model_name, temperature=temperature, **kwargs)
    elif llm == 'ollama':
        # https://python.langchain.com/v0.1/docs/integrations/chat/ollama/
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=model_name)  # e.g., 'llama3'
    else:
        # add any LLMs you want to use here using LangChain
        raise NotImplementedError(f"LLM '{llm}' not implemented yet.")
