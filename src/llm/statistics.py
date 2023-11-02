from collections import defaultdict
from dataclasses import field
from decimal import Decimal

from pydantic import BaseModel, computed_field

from src.llm.enums import ChatModel


class TokenPricing(BaseModel):
    """Tracks prices in USD per 1000 tokens"""

    prompt_1k: Decimal
    completion_1k: Decimal


class LlmPricing(BaseModel):
    chat_model_pricing: dict[ChatModel, TokenPricing]

    def completion_1k(self, model_name: ChatModel) -> Decimal:
        try:
            return self.chat_model_pricing[model_name].completion_1k
        except KeyError as exc:
            raise KeyError(f"Model {model_name} not found in pricing info") from exc

    def prompt_1k(self, model_name: ChatModel) -> Decimal:
        try:
            return self.chat_model_pricing[model_name].prompt_1k
        except KeyError as exc:
            raise KeyError(f"Model {model_name} not found in pricing info") from exc


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LlmStatistics(BaseModel):
    token_usage: dict[ChatModel, TokenUsage] = field(default_factory=lambda: defaultdict(TokenUsage))

    @computed_field
    @property
    def total_prompt_tokens(self) -> int:
        return sum(token_usage.prompt_tokens for token_usage in self.token_usage.values())

    @computed_field
    @property
    def total_completion_tokens(self) -> int:
        return sum(token_usage.completion_tokens for token_usage in self.token_usage.values())

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @computed_field
    @property
    def total_cost(self) -> Decimal:
        cost = Decimal(0)
        for model_name, token_usage in self.token_usage.items():
            model = ChatModel(model_name)
            cost += token_usage.prompt_tokens * _LLM_PRICING.prompt_1k(model) * Decimal(0.001)
            cost += token_usage.completion_tokens * _LLM_PRICING.completion_1k(model) * Decimal(0.001)

        return cost

    def update(self, model: ChatModel, prompt_tokens: int, completion_tokens: int) -> None:
        self.token_usage[model].prompt_tokens += prompt_tokens
        self.token_usage[model].completion_tokens += completion_tokens


_LLM_PRICING = LlmPricing(
    chat_model_pricing={
        ChatModel.GPT_35: TokenPricing(
            prompt_1k=Decimal(0.001),
            completion_1k=Decimal(0.002),
        ),
        ChatModel.GPT_4: TokenPricing(
            prompt_1k=Decimal(0.03),
            completion_1k=Decimal(0.06),
        ),
        ChatModel.LLAMA_7B: TokenPricing(
            prompt_1k=Decimal(0.00015),
            completion_1k=Decimal(0.00015),
        ),
        ChatModel.LLAMA_13B: TokenPricing(
            prompt_1k=Decimal(0.00025),
            completion_1k=Decimal(0.00025),
        ),
        ChatModel.LLAMA_70B: TokenPricing(
            prompt_1k=Decimal(0.001),
            completion_1k=Decimal(0.001),
        ),
    }
)
