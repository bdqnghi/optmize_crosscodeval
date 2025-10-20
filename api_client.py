import requests
import json
from typing import Optional, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential


class ModelAPIClient:
    """Client for interacting with Qwen/Qwen2.5-Coder models."""

    def __init__(
        self,
        model_size: str = "3b",
        config_path: str = "config.json"
    ):
        """
        Initialize API client with specified model size.

        Args:
            model_size: Model size to use ("3b", "7b", or "14b")
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        if model_size not in config["models"]:
            raise ValueError(f"Model size '{model_size}' not found in config. Available: {list(config['models'].keys())}")

        model_config = config["models"][model_size]
        self.base_url = model_config["api_base"]
        self.api_key = model_config["api_key"]
        self.model = model_config["name"]
        self.model_size = model_size

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(
        self,
        prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        n: int = 1,
        top_p: Optional[float] = None
    ) -> List[str]:
        """
        Generate code completion from the model.

        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            n: Number of completions to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)

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

        if top_p is not None:
            payload["top_p"] = top_p

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
