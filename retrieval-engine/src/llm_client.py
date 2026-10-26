import os
import logging
import requests

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model: str, api_key_env: str, timeout: float = 60.0, max_tokens: int = 400):
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Переменная {api_key_env} не задана")

        self.model, self.timeout = model, timeout
        if max_tokens <= 0:
            raise ValueError('max_tokens must be positive.')
        self.max_tokens = max_tokens
        self.url = "https://openrouter.ai/api/v1/chat/completions"

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
        })
 
    def generate(self, messages: list[dict[str, str]]) -> str:
        message = self._request(messages)
        answer = message.get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Model returned an empty answer")
        return answer

    def generate_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """Return the assistant message, including tool calls.
        """

        message = self._request(messages, tools=tools)
        content = message.get("content")

        if not message.get("tool_calls") and (
            not isinstance(content, str) or not content.strip()
        ):
            raise RuntimeError("Model returned neither text nor tool calls")
        return message

    def _request(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        
        if tools is not None:
            payload["tools"] = tools

        try:
            response = self.session.post(
                self.url, json=payload, timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("OpenRouter request failed (%s)", type(exc).__name__)
            raise

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("OpenRouter returned invalid JSON")
            raise RuntimeError("OpenRouter returned invalid JSON") from exc

        if isinstance(data, dict) and "error" in data:
            logger.error("OpenRouter returned an API error")
            raise RuntimeError("OpenRouter returned an API error")

        try:
            choice = data["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError("Invalid assistant message")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error("OpenRouter returned an invalid response structure")
            raise RuntimeError("OpenRouter returned an invalid response structure") from exc

        if choice.get("finish_reason") == "length":
            logger.warning("Model response reached the output token limit")

        return message
