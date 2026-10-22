import os
import requests

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
        response = self.session.post(
            self.url,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": self.max_tokens,
            },
            timeout=self.timeout,
        )

        if not response.ok:
            print("OpenRouter:", response.text)

        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise RuntimeError("OpenRouter returned error.")

        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter returned error") from exc

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Model returned an empty answer")

        if data['choices'][0].get('finish_reason') == 'length':
            print('Warning: answer reached the output token limit and may be incomplete.')

        return answer
