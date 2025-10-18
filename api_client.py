import requests
import json
from typing import Optional, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential


class ModelAPIClient:
    """Client for interacting with the Qwen/Qwen2.5-Coder-3B model API."""

    def __init__(
        self,
        base_url: str = "http://195.201.127.59/v1/completions",
        api_key: str = "zen8labs",
        model: str = "Qwen/Qwen2.5-Coder-3B"
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        n: int = 1
    ) -> List[str]:
        """
        Generate code completion from the model.

        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            n: Number of completions to generate

        Returns:
            List of generated completions
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n
        }

        if stop:
            payload["stop"] = stop

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            completions = [choice["text"] for choice in result["choices"]]
            return completions

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            raise

    def generate_fim(
        self,
        prefix: str,
        suffix: str,
        template: str,
        stop_tokens: List[str],
        temperature: float = 0.01,
        max_tokens: int = 512,
        n: int = 1
    ) -> List[str]:
        """
        Generate Fill-In-the-Middle (FIM) completion.

        Args:
            prefix: Code before the cursor
            suffix: Code after the cursor
            template: FIM template with {prefix}, {suffix} placeholders
            stop_tokens: List of stop sequences for FIM
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            n: Number of completions to generate

        Returns:
            List of generated middle sections
        """
        prompt = template.format(prefix=prefix, suffix=suffix)
        return self.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_tokens,
            n=n
        )
