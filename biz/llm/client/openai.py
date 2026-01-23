import os
from typing import Dict, List, Optional

from openai import OpenAI

from biz.llm.client.base import BaseClient
from biz.llm.types import NotGiven, NOT_GIVEN


class OpenAIClient(BaseClient):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com")
        if not self.api_key:
            raise ValueError("API key is required. Please provide it or set it in the environment variables.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.default_model = os.getenv("OPENAI_API_MODEL", "gpt-5.2")

    def completions(self,
                    messages: List[Dict[str, str]],
                    model: Optional[str] | NotGiven = NOT_GIVEN,
                    ) -> str:
        model = model or self.default_model
        stream = self.client.responses.create(
            model=model,
            input=messages,
            reasoning={"effort": "high"},
            stream=True,
        )

        full_text = ""
        for chunk in stream:
            if chunk.type == "response.completed":
                full_text = chunk.response.output_text or ""
                break
            
            elif chunk.type == "response.output_text.delta":
                full_text += chunk.delta or ""

        # 如果没等到 completed，也用拼接的（fallback）
        return full_text.strip()
