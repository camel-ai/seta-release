from enum import Enum
from camel.agents import ChatAgent
# from __future__ import annotations

import asyncio
import atexit
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)

from pydantic import BaseModel

from camel.agents.base import BaseAgent
from camel.logger import get_logger

from camel.messages import (
    BaseMessage,
    FunctionCallingMessage,
    OpenAIMessage,
)
from camel.models import ModelFactory
from camel.prompts import TextPrompt
from camel.responses import ChatAgentResponse
from camel.toolkits import FunctionTool, NoteTakingToolkit
from camel.types import (
    OpenAIBackendRole,
    RoleType
)
from camel.types.agents import ToolCallingRecord

logger = get_logger(__name__)

# Cleanup temp files on exit
_temp_files: Set[str] = set()
_temp_files_lock = threading.Lock()


def _cleanup_temp_files():
    with _temp_files_lock:
        for path in _temp_files:
            try:
                os.unlink(path)
            except Exception:
                pass


atexit.register(_cleanup_temp_files)


try:
    from seta_env.utils.perf_tracer import PerfTracer as _PerfTracer
except ImportError:
    _PerfTracer = None  # type: ignore[assignment]

try:
    from areal.utils import perf_tracer
    from areal.utils.perf_tracer import (
        atrace_session_phase,
        session_context,
        trace_perf,
        trace_session,
        Category,
        atrace_scope
    )
except ImportError:
    # If areal is not installed, use dummy implementations
    import contextlib
    
    @contextlib.asynccontextmanager
    async def atrace_scope(*args, **kwargs):
        yield
    
    @contextlib.asynccontextmanager  
    async def atrace_session_phase(*args, **kwargs):
        yield
    
    def session_context(*args, **kwargs):
        return contextlib.nullcontext()
    
    def trace_perf(*args, **kwargs):
        pass
    
    def trace_session(*args, **kwargs):
        pass
    
    # Mock Category with needed attributes
    class Category:
        COMM = "COMM"
        IO = "IO"

# Enum for finish reason
class TerminationReason(Enum):
    # max_parse_errors reached
    MAX_PARSE_ERRORS = "max_parse_errors"
    # max_iteration reached
    MAX_ITERATION_REACHED = "max_iteration_reached"
    # step timeout
    STEP_TIMEOUT = "step_timeout"
    # max_tokens reached
    MAX_TOKENS_EXCEEDED = "max_tokens_exceeded"
    # task finishes
    TASK_FINISHED = "task_finished"
    # exceeds completion length
    COMPLETION_LENGTH_EXCEEDED = "completion_length_exceeded"
    # unknown error
    UNKNOWN_ERROR = "unknown_error"
    # Not set
    NOT_SET = "not_set"


from camel.terminators import TokenLimitTerminator, ResponseWordsTerminator


class parse_error_check():

    def __init__(
                self, 
                bot_token = "<tool_call>\n",
                eot_token = "\n</tool_call>",
                tokenizer = None,
                max_parse_error = 3
                ):
        self.parse_error_count = 0
        self.bot_token = bot_token
        self.eot_token = eot_token
        self.max_parse_error = max_parse_error
        # if tokenizer, extract bot_token and eot_token from tokenizer
    
    def reset(self):
        self.parse_error_count = 0
    
    async def check(self, response):
        try:
            content = response.output_messages[0].content
        except Exception as e:
            logger.error(f"No content found in response: {e}")
            return True

        if (not content) or ((self.bot_token not in content) and (self.eot_token not in content)):
            return None
        
        # Find all potential tool call blocks
        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            return None

        # Check each match for JSON parse errors
        # Note: We return after finding the first error to allow iterative correction
        for match_text in matches:
            try:
                # Try to parse the JSON
                json.loads(match_text.strip())
                # If successful, no error for this match
                continue
            except json.JSONDecodeError as e:
                # Found a parse error - handle it and return (one error at a time)
                self.parse_error_count += 1
                logger.warning(
                    f"Detected JSON parse error (count: {self.parse_error_count}/{self.max_parse_error}): {str(e)}"
                )
                logger.warning(f"Problematic content: {match_text[:200]}...")
                
                # Create an error tool calling record
                error_message = (
                    f"JSON Parse Error: {str(e)}\n"
                    f"The tool call format is incorrect. Please ensure:\n"
                    f"1. The JSON is valid and properly formatted\n"
                    f"2. All quotes are properly escaped\n"
                    f"3. The structure matches: {{'name': 'function_name', 'arguments': {{}}}}\n"
                    f"Problematic content (first 200 chars): {match_text[:200]}..."
                )

                continue_message = BaseMessage.make_user_message(
                                role_name="User", content=error_message
                            )
                # early return 
                return continue_message, self.parse_error_count==self.max_parse_error
            
        return None, 0



class AgentTrain(ChatAgent):
    """A ChatAgent with performance tracing capabilities."""
    def __init__(self, task_name: str, *args, **kwargs):
        """Initialize ChatAgentTrace with parse error tracking."""
        super().__init__(*args, **kwargs)
        self.max_parse_errors = kwargs.get('max_parse_errors', 10)
        self.parse_error_count = 0
        self.task_name = task_name
        self.termination_reason = TerminationReason.NOT_SET
        self.summary_window_ratio = None

        # TODO parse the bot_token and eot_token from the tokenizer if provided, to support different formats
        # tokenzier = self.model.tokenizer if hasattr(self.model, 'tokenizer') else None
        # if tokenzier:

        self.parse_error_checker = parse_error_check()

        # we want to record important information for training debugging
        self.meta_info_record = {
            "iteration_count": 0,
            "termination_reason": TerminationReason.NOT_SET,
            "max_parallel_tool_call": 0,
            "parse_error_count": 0,
            "total_tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def reset(self):
        """Reset the agent's state for a new task."""
        super().reset()
        self.parse_error_checker.reset()
        self.termination_reason = TerminationReason.NOT_SET

    
    async def _astep_non_streaming_task(
        self,
        input_message: Union[BaseMessage, str],
        response_format: Optional[Type[BaseModel]] = None,
    ) -> ChatAgentResponse:
        r"""Internal async method for non-streaming astep logic.
        
        1. Record initial input prompt in memory
        2. Main iteration loop for agent response generation
            a. Get context from memory
            b. Request model response
            c. Tool call pass error check
            d. Tool call iteration, if no tool call, returns
            e. Terminator check (token limit check)
        

        What to record and return

            prompt
            |
            |
        agent memory --> model backend --> log request 
                                            |
                                            |
                                            client
                                            |
                                            |
        record      <-- model backend <--log request 
        assistant 
        tool cal
        in memory
            |
            |
        tool exec
        record tool result
        in memory
            |
            |
        

        """
        from camel.utils.agent_context import set_current_agent_id

        set_current_agent_id(self.agent_id)

        try:
            from camel.utils.langfuse import set_current_agent_session_id

            set_current_agent_session_id(self.agent_id)
        except ImportError:
            pass  # Langfuse not available

        # Check if this call is from a RegisteredAgentToolkit to prevent tool
        # use
        disable_tools = self._is_called_from_registered_toolkit()

        # Handle response format compatibility with non-strict tools
        original_response_format = response_format
        input_message, response_format, used_prompt_formatting = (
            self._handle_response_format_with_non_strict_tools(
                input_message, response_format
            )
        )

        if isinstance(input_message, str):
            input_message = BaseMessage.make_user_message(
                role_name="User", content=input_message
            )

        self.update_memory(input_message, OpenAIBackendRole.USER)

        tool_call_records: List[ToolCallingRecord] = []
        external_tool_call_requests: Optional[List[ToolCallRequest]] = None
        accumulated_context_tokens = (
            0  # This tracks cumulative context tokens, not API usage tokens
        )

        # Initialize token usage tracker
        step_token_usage = self._create_token_usage_tracker()
        iteration_count: int = 0
        prev_num_openai_messages: int = 0

        # Track if we've recorded tool calls for the current response
        # to avoid duplicate assistant message recording
        recorded_tool_calls = False

        # Reset meta_info_record for this call; fields are updated inline as
        # they change so the record reflects current state at any exit point
        self.meta_info_record = {
            "iteration_count": 0,
            "termination_reason": TerminationReason.NOT_SET,
            "max_parallel_tool_call": 0,
            "parse_error_count": 0,
            "total_tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Grab the per-trajectory perf tracer if one was attached by
        # TerminalEnvironment (attribute is None / absent when running outside
        # that framework).
        _tracer = getattr(self, '_perf_tracer', None)

        while True:
            # ── iteration-level span ──────────────────────────────────────
            _iter_n = iteration_count + 1  # human-readable (1-based)
            if _tracer:
                _tracer.begin(
                    "iteration", cat="agent",
                    tid=_tracer.TID_AGENT,
                    args={"n": _iter_n},
                )

            if self.pause_event is not None and not self.pause_event.is_set():
                if isinstance(self.pause_event, asyncio.Event):
                    await self.pause_event.wait()
                elif isinstance(self.pause_event, threading.Event):
                    # For threading.Event in async context, run in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.pause_event.wait)
            try:
                (
                    openai_messages,
                    num_tokens,
                ) = await self._get_context_with_summarization_async()
                accumulated_context_tokens += num_tokens
            except RuntimeError as e:
                self.termination_reason = TerminationReason.MAX_TOKENS_EXCEEDED
                self.meta_info_record["termination_reason"] = TerminationReason.MAX_TOKENS_EXCEEDED
                if _tracer:
                    _tracer.end("iteration", cat="agent", tid=_tracer.TID_AGENT,
                                args={"stop": "max_tokens_exceeded"})
                return self._step_terminate(
                    e.args[1], tool_call_records, "max_tokens_exceeded"
                )

            # No trimming: terminate when context exceeds token limit.
            if self._token_limit is not None and num_tokens > self._token_limit:
                self.termination_reason = TerminationReason.MAX_TOKENS_EXCEEDED
                self.meta_info_record["termination_reason"] = TerminationReason.MAX_TOKENS_EXCEEDED
                if _tracer:
                    _tracer.end("iteration", cat="agent", tid=_tracer.TID_AGENT,
                                args={"stop": "max_tokens_exceeded"})
                return self._step_terminate(
                    num_tokens, tool_call_records, "max_tokens_exceeded"
                )

            # ── model request span ────────────────────────────────────────
            if _tracer:
                _tracer.begin(
                    "model_request", cat="model",
                    tid=_tracer.TID_MODEL,
                    args={"iteration": _iter_n, "context_tokens": num_tokens},
                )

            # Get response from model backend
            response = await self._aget_model_response(
                openai_messages,
                current_iteration=iteration_count,
                response_format=response_format,
                tool_schemas=[]
                if disable_tools
                else self._get_full_tool_schemas(),
                prev_num_openai_messages=prev_num_openai_messages,
            )

            if _tracer:
                _usage = response.usage_dict or {}
                _tracer.end(
                    "model_request", cat="model",
                    tid=_tracer.TID_MODEL,
                    args={
                        "prompt_tokens":     _usage.get("prompt_tokens"),
                        "completion_tokens": _usage.get("completion_tokens"),
                        "total_tokens":      _usage.get("total_tokens"),
                        "finish_reason":     str(response.finish_reasons[0])
                                             if getattr(response, "finish_reasons", None)
                                             else None,
                    },
                )

            prev_num_openai_messages = len(openai_messages)
            iteration_count += 1
            self.meta_info_record["iteration_count"] = iteration_count

            # Accumulate API token usage
            self._update_token_usage_tracker(
                step_token_usage, response.usage_dict
            )
            self.meta_info_record["prompt_tokens"] = step_token_usage["prompt_tokens"]
            self.meta_info_record["completion_tokens"] = step_token_usage["completion_tokens"]
            self.meta_info_record["total_tokens"] = step_token_usage["total_tokens"]
            await self._aemit_request_usage(
                usage_dict=response.usage_dict,
                step_usage=step_token_usage.copy(),
                request_index=iteration_count,
                response_id=response.response_id,
            )

            # Update token cache from LLM response
            self._update_token_cache(response.usage_dict, len(openai_messages))

            # Terminate Agent if stop_event is set
            if self.stop_event and self.stop_event.is_set():
                # Use the _step_terminate to terminate the agent with reason
                logger.info(
                    f"Termination triggered at iteration {iteration_count}"
                )
                self.termination_reason = TerminationReason.STEP_TIMEOUT
                self.meta_info_record["termination_reason"] = TerminationReason.STEP_TIMEOUT
                if _tracer:
                    _tracer.end("iteration", cat="agent", tid=_tracer.TID_AGENT,
                                args={"stop": "step_timeout"})
                return self._step_terminate(
                    accumulated_context_tokens,
                    tool_call_records,
                    "termination_triggered",
                )

            # Reset flag for each iteration
            recorded_tool_calls = False

            if tool_call_requests := response.tool_call_requests:
                # Track the largest batch of parallel tool calls seen this step
                self.meta_info_record["max_parallel_tool_call"] = max(
                    self.meta_info_record["max_parallel_tool_call"],
                    len(tool_call_requests),
                )

                # Separate internal and external tool calls
                internal_tool_requests = []
                for tool_call_request in tool_call_requests:
                    if (
                        tool_call_request.tool_name
                        in self._external_tool_schemas
                    ):
                        if external_tool_call_requests is None:
                            external_tool_call_requests = []
                        external_tool_call_requests.append(tool_call_request)
                    else:
                        internal_tool_requests.append(tool_call_request)

                # Record the assistant message with ALL tool calls (internal +
                # external) BEFORE executing any tools.
                response_content = ""
                if response.output_messages:
                    response_content = (
                        response.output_messages[0].content or ""
                    )
                self._record_assistant_tool_calls_from_requests(
                    tool_call_requests, content=response_content
                )
                recorded_tool_calls = True

                # Execute internal tools only
                for tool_call_request in internal_tool_requests:
                    if (
                        self.pause_event is not None
                        and not self.pause_event.is_set()
                    ):
                        if isinstance(self.pause_event, asyncio.Event):
                            await self.pause_event.wait()
                        elif isinstance(self.pause_event, threading.Event):
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(
                                None, self.pause_event.wait
                            )
                    if _tracer:
                        _tracer.begin(
                            "tool_call", cat="tool",
                            tid=_tracer.TID_TOOL,
                            args={"tool_name": tool_call_request.tool_name,
                                  "iteration": iteration_count},
                        )
                    tool_call_record = await self._aexecute_tool(
                        tool_call_request
                    )
                    if _tracer:
                        _tracer.end("tool_call", cat="tool", tid=_tracer.TID_TOOL)
                    tool_call_records.append(tool_call_record)
                    self.meta_info_record["total_tool_calls"] = len(tool_call_records)

                # If we found an external tool call, break the loop
                if external_tool_call_requests:
                    self.termination_reason = TerminationReason.TASK_FINISHED
                    self.meta_info_record["termination_reason"] = TerminationReason.TASK_FINISHED
                    if _tracer:
                        _tracer.end("iteration", cat="agent", tid=_tracer.TID_AGENT,
                                    args={"stop": "task_finished_external_tool"})
                    break

                # Check for JSON parse errors in text-format tool calls (e.g.
                # Qwen-style models that embed tool calls in content as
                # <tool_call>...</tool_call>).  After valid tools are executed,
                # feed the error back as a user message so the model can fix
                # the malformed call on the next iteration.
                parse_result = await self.parse_error_checker.check(response)
                if isinstance(parse_result, tuple) and parse_result[0] is not None:
                    continue_message, is_max_error = parse_result
                    self.meta_info_record["parse_error_count"] = self.parse_error_checker.parse_error_count
                    if is_max_error:
                        self.termination_reason = TerminationReason.MAX_PARSE_ERRORS
                        self.meta_info_record["termination_reason"] = TerminationReason.MAX_PARSE_ERRORS
                        if _tracer:
                            _tracer.end("iteration", cat="agent", tid=_tracer.TID_AGENT,
                                        args={"stop": "max_parse_errors"})
                        return self._step_terminate(
                            accumulated_context_tokens,
                            tool_call_records,
                            "max_parse_errors",
                        )
                    self.update_memory(continue_message, OpenAIBackendRole.USER)
                    if _tracer:
                        _tracer.end("iteration", cat="agent", tid=_tracer.TID_AGENT,
                                    args={"stop": "parse_error_continue"})
                    continue

                if (
                    self.max_iteration is not None
                    and iteration_count >= self.max_iteration
                ):
                    self.termination_reason = TerminationReason.MAX_ITERATION_REACHED
                    self.meta_info_record["termination_reason"] = TerminationReason.MAX_ITERATION_REACHED
                    if _tracer:
                        _tracer.end("iteration", cat="agent", tid=_tracer.TID_AGENT,
                                    args={"stop": "max_iteration"})
                    break

                if _tracer:
                    _tracer.end("iteration", cat="agent", tid=_tracer.TID_AGENT)
                continue

            # No tool calls — model decided to stop
            self.termination_reason = TerminationReason.TASK_FINISHED
            self.meta_info_record["termination_reason"] = TerminationReason.TASK_FINISHED
            if _tracer:
                _tracer.end("iteration", cat="agent", tid=_tracer.TID_AGENT,
                            args={"stop": "task_finished"})
            break

        await self._aformat_response_if_needed(response, response_format)

        # Apply manual parsing if we used prompt-based formatting
        if used_prompt_formatting and original_response_format:
            self._apply_prompt_based_parsing(
                response, original_response_format
            )

        # Only record final output if we haven't already recorded tool calls
        # for this response (to avoid duplicate assistant messages)
        if not recorded_tool_calls:
            self._record_final_output(response.output_messages)

        # Clean tool call messages from memory after response generation
        if self.prune_tool_calls_from_memory and tool_call_records:
            self.memory.clean_tool_calls()

        return self._convert_to_chatagent_response(
            response,
            tool_call_records,
            accumulated_context_tokens,
            external_tool_call_requests,
            step_token_usage["prompt_tokens"],
            step_token_usage["completion_tokens"],
            step_token_usage["total_tokens"],
        )
