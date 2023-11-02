from enum import StrEnum


class ChatModel(StrEnum):
    GPT_4 = "gpt-4"
    GPT_35 = "gpt-3.5-turbo-0613"

    LLAMA_7B = "meta-llama/Llama-2-7b-chat-hf"
    LLAMA_13B = "meta-llama/Llama-2-13b-chat-hf"
    LLAMA_70B = "meta-llama/Llama-2-70b-chat-hf"
