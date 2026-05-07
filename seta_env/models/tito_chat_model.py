"""TITO (Turn-In, Turn-Out) Chat Model Backend.

Inherits from OpenAICompatibleModel and maintains a per-session message
history using raw API responses (via model_dump), preserving all fields
including reasoning_content. This ensures the model sees its prior reasoning
on subsequent turns and improves KV cache hit rates.
"""

from typing import Any, Dict, List, Optional, Type, Union

from openai import AsyncStream
from openai.lib.streaming.chat import AsyncChatCompletionStreamManager
from pydantic import BaseModel

from camel.logger import get_logger
from camel.messages import OpenAIMessage
from camel.models.openai_compatible_model import OpenAICompatibleModel
from camel.types import ChatCompletion, ChatCompletionChunk
from camel.utils import BaseTokenCounter

logger = get_logger(__name__)


class TITOChatModel(OpenAICompatibleModel):
    """OpenAI-compatible model that preserves raw API responses across turns.

    Maintains its own session message list, storing raw model_dump()
    responses to preserve all fields (reasoning_content, tool_calls, etc.).
    On each call, it replaces the incoming agent messages with its own
    session history so the API sees the full, lossless conversation.

    Extra __init__ kwargs (popped before super()):
        tito_validate (bool): If True, log validation comparisons between
            agent messages and TITO session messages. Default False.
    """

    def __init__(self, *args, **kwargs):
        self._tito_validate = kwargs.pop("tito_validate", False)
        super().__init__(*args, **kwargs)
        self._session_messages: List[Dict[str, Any]] = []
        self._cache_stats: Dict[str, Any] = {
            "total_prompt_tokens": 0,
            "total_cached_tokens": 0,
            "per_turn_usage": [],
        }

    def preprocess_messages(
        self, messages: List[OpenAIMessage]
    ) -> List[OpenAIMessage]:
        """No-op: TITO manages its own raw session messages.

        The default preprocess_messages strips <think> tags and reorders
        tool calls. TITO bypasses this so that:
        1. _arun receives the unmodified agent messages (needed to extract
           new tool results on subsequent turns).
        2. The TITO session messages — which already contain raw content
           from model_dump() — are sent directly to the API without
           stripping reasoning content.
        """
        return messages

    def reset(self):
        """Reset session state for a new episode."""
        self._session_messages = []
        self._cache_stats = {
            "total_prompt_tokens": 0,
            "total_cached_tokens": 0,
            "per_turn_usage": [],
        }

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[
        ChatCompletion,
        AsyncStream[ChatCompletionChunk],
        AsyncChatCompletionStreamManager[BaseModel],
    ]:
        """Core TITO logic: maintain session messages with raw responses."""
        if not self._session_messages:
            # First call in episode: store all messages (system + user prompt)
            self._session_messages = [dict(m) for m in messages]
        else:
            # Extract new messages: scan backward to find last assistant msg
            new_msgs = self._extract_new_messages(messages)
            self._session_messages.extend(new_msgs)

        # Optional validation
        if self._tito_validate:
            self._log_validation(messages)

        # Call parent _arun with our lossless session messages
        response = await super()._arun(
            self._session_messages, response_format, tools
        )

        # Record raw response -- model_dump() preserves everything
        raw_msg = self._capture_raw_response(response)
        self._session_messages.append(raw_msg)

        # Track cache stats
        self._update_cache_stats(response)

        return response

    def _extract_new_messages(
        self, messages: List[OpenAIMessage]
    ) -> List[Dict[str, Any]]:
        """Backward scan to find messages after the last assistant message."""
        last_asst_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_asst_idx = i
                break

        if last_asst_idx == -1:
            logger.warning(
                "TITO: no assistant message found, resetting session"
            )
            self._session_messages = []
            return [dict(m) for m in messages]

        # Everything after the last assistant = new tool/user messages
        new_msgs = messages[last_asst_idx + 1 :]
        return [dict(m) for m in new_msgs if m.get("role") != "assistant"]

    def _capture_raw_response(
        self, response: ChatCompletion
    ) -> Dict[str, Any]:
        """Use model_dump() on the response message to preserve ALL fields."""
        choice = response.choices[0]
        msg = choice.message.model_dump(exclude_none=True)
        msg["role"] = "assistant"
        # Clean up tool_calls: remove 'index' field that some backends
        # (e.g. sglang) add with value -1 — it's not part of the OpenAI
        # chat API schema and can cause issues on subsequent requests.
        if "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                tc.pop("index", None)
        return msg

    def _update_cache_stats(self, response: ChatCompletion) -> None:
        """Extract cached_tokens from usage.prompt_tokens_details."""
        if not response.usage:
            return
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        self._cache_stats["total_prompt_tokens"] += prompt_tokens
        self._cache_stats["total_cached_tokens"] += cached
        self._cache_stats["per_turn_usage"].append(
            {
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached,
                "completion_tokens": completion_tokens,
            }
        )

    def _log_validation(self, agent_messages: List[OpenAIMessage]) -> None:
        """Compare agent messages with TITO session messages for debugging."""
        agent_count = len(agent_messages)
        tito_count = len(self._session_messages)

        if agent_count != tito_count:
            logger.info(
                f"TITO validation: count mismatch - "
                f"agent={agent_count}, tito={tito_count}"
            )

        for i, tito_msg in enumerate(self._session_messages):
            if tito_msg.get("role") == "assistant" and tito_msg.get(
                "reasoning_content"
            ):
                if i < agent_count:
                    agent_msg = agent_messages[i]
                    if not agent_msg.get("reasoning_content"):
                        logger.info(
                            f"TITO: preserved reasoning_content "
                            f"at position {i}"
                        )

        agent_roles = [m.get("role", "?") for m in agent_messages]
        tito_roles = [m.get("role", "?") for m in self._session_messages]
        if agent_roles != tito_roles:
            logger.debug(
                f"TITO validation: role sequences differ - "
                f"agent={agent_roles}, tito={tito_roles}"
            )
