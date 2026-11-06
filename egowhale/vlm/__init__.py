"""Shared VLM plumbing for coarse annotation and the L3 audit."""

from egowhale.vlm.client import MockClient, OpenAICompatibleClient, VLMClient, VLMError
from egowhale.vlm.config import VLMConfig

__all__ = ["MockClient", "OpenAICompatibleClient", "VLMClient", "VLMConfig", "VLMError"]
