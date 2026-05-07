"""AgentTrain with TITO model backend support and cache stats."""

from typing import Optional, Type, Union

from pydantic import BaseModel

from camel.messages import BaseMessage
from camel.responses import ChatAgentResponse

from seta_env.agent.train_agent import AgentTrain


class AgentTrainTITO(AgentTrain):
    """AgentTrain with TITO model backend support and cache stats.

    Propagates reset() to the TITO model backend and extracts cache
    statistics into meta_info_record after each step.
    """

    def reset(self):
        """Reset the agent and TITO model session for a new episode."""
        super().reset()
        if hasattr(self, "model_backend") and hasattr(
            self.model_backend, "reset"
        ):
            self.model_backend.reset()

    async def _astep_non_streaming_task(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
    ) -> ChatAgentResponse:
        # Call parent implementation
        result = await super()._astep_non_streaming_task(
            input_message, response_format
        )

        # Extract TITO cache stats into meta_info_record
        if hasattr(self, "model_backend") and hasattr(
            self.model_backend, "_cache_stats"
        ):
            stats = self.model_backend._cache_stats
            self.meta_info_record["cached_tokens"] = stats.get(
                "total_cached_tokens", 0
            )
            total_prompt = stats.get("total_prompt_tokens", 1)
            self.meta_info_record["cache_hit_ratio"] = stats[
                "total_cached_tokens"
            ] / max(1, total_prompt)
            self.meta_info_record["per_turn_cache"] = stats.get(
                "per_turn_usage", []
            )

        return result
