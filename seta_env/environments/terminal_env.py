# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, Tuple

from pydantic import BaseModel, Field

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from camel.messages import BaseMessage

from seta_env.utils.utils import async_timer, load_main_trajectory
from seta_env.utils.perf_tracer import PerfTracer
from seta_env.agent.train_agent import AgentTrain as _DefaultAgent, NoteTakingToolkit
from seta_env.agent.prompt_loader import get_agent_class, load_system_message
from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime
from seta_env.verifiers.verifier import Verifier
from seta_env.verifiers.reward_fn import reward_factory


_IMPORTANT_TERMINATION_REASONS = {
    "task_finished",
    "max_iteration_reached",
    "max_tokens_exceeded",
    "completion_length_exceeded",
}

class TerminalEnvironment:
    def __init__(self,
                 agent_config: Dict[str, Any],
                 model_config: Dict[str, Any],
                 runtime_config: Dict[str, Any],
                 env_config: Dict[str, Any],
                 perf_tracer: Optional["PerfTracer"] = None,
                 ):
        r"""Initializes the TerminalEnvironment.
        
        The environment is a wrapper around agent execution
        in a terminal runtime. It manages the lifecycle of the agent, including
        resetting the environment, running the agent, evaluating its performance,
        calculating rewards, and closing the environment after execution.

        necessary components:
            - modelbackend: interface with LLM model
            - runtime: terminal docker container runtime, 
            - toolkits: toolkits for agent to use, e.g., notetaking, terminal toolkit
            - agent: train_agent class that runs on the runtime
            - verifier: verifier class that evaluates final states of the runtime
                        when agent finishes execution
            - reward_fn: reward function that calculates reward based on evaluation results

        Args:
            agent_config (Dict[str, Any]):
                Configuration example:{
                            ""
                }

            model_config (Dict[str, Any]):
                Configuration example:{
                    "model": ModelBackend instance,
                }
            
            runtime_config (Dict[str, Any]):
                Configuration example:{
                    "task_dir": str, # path to task folder, which contains task.toml, instruction.md, etc.
                    "trial_root": str, # root path to store trial outputs,
                    "session_id": str, # unique identifier for the trial session, aka trial_name, docker container name, etc.
                    "environment_type": str, # "docker", "daytona", "modal", etc.
                    # "environment": BaseEnvironment instance, # optional, if provided, environment_type will be ignored and this environment will be used directly
                }
            env_config (Dict[str, Any]):
                Configuration example:{
                    "reward_fn": str, # reward function name, e.g., "pass_ratio",
                    "task_timeouts": dict, # timeouts for different stages of the task, e.g., {"total_timeout": 3600, "per_step_timeout": 300}
                }
            
        """

        self.agent_config   = agent_config
        self.model_config   = model_config
        self.runtime_config = runtime_config
        self.env_config     = env_config
        self.reward_fn      = env_config.get("reward_fn", "pass_ratio")
        # Externally supplied tracer (e.g. shared across envs); None means
        # TerminalEnvironment creates its own tracer per step() call.
        self._perf_tracer_ext: Optional[PerfTracer] = perf_tracer
    
    async def step(self, task: dict, uid: str, traj_i: int=0) -> dict:
        r"""
        Execute a complete agent workflow:
        1. Reset runtime (with toolkit)
        2. Reset agent
        3. Agent run
        4. Evaluation
        5. Cleanup

        Args:
            task (dict): {
                "task_name": str,
                "task_path": str, # path to task folder, which contains task.toml, instruction.md, etc.
                "instruction": str, # task instruction
            }

        Returns:
            reward (float): The calculated reward for the agent's performance on the given task.
            run_info (dict):
                Running results and logs
                task_id
                uid
                traj_i
                reward
                timings: {}
                trajectory: []
                evaluation: {"test1": 0/1, "test2": 0/1, ...}
                error_info: empty if no error,
                    else: {"stage": "stage name where error happens", 
                            "error_message": "error message"}
                agent_summary: {
                    "num_iterations": int,
                    "num_parse_error": int,
                    "stop_reason": "reason for agent to stop, e.g., max_iteration, finish_tool_use, parse_error",
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "total_tokens": int,
                }

        """
        self.task_name = task.get("task_name")
        self.task = task  # Store for oracle agent to access task_path
        self.uid = uid
        self.traj_i = traj_i
        self.error_info = {}
        self.agent_summary = {}
        self.evaluation_results = {}
        reward = None

        # One PerfTracer per trajectory; tracks are: env / agent / model / tools.
        # Use an externally supplied tracer if provided, otherwise create one
        # whose session_id is "<task_name>_<uid>" for easy identification.
        if self._perf_tracer_ext is not None:
            self._perf_tracer = self._perf_tracer_ext
        else:
            _session_id = f"{self.task_name}_{uid}" if self.task_name else uid
            self._perf_tracer = PerfTracer(session_id=_session_id)

        stage_timings = {}
        task_timeouts = self.env_config.get("task_timeouts", {})
        try:
            current_stage = "1_reset_env"
            async with self._perf_tracer.span(current_stage, cat="env"):
                async with async_timer(
                    current_stage,
                    stage_timings,
                    timeout=task_timeouts.get("_reset_env"),
                ):
                    await self._reset_env(task, uid)

            current_stage = "2_run_agent"
            async with self._perf_tracer.span(current_stage, cat="env"):
                async with async_timer(
                    current_stage,
                    stage_timings,
                    timeout=task_timeouts.get("agent_astep"),
                ):
                    await self.run_agent()

            current_stage = "3_evaluate"
            async with self._perf_tracer.span(current_stage, cat="env"):
                async with async_timer(
                    current_stage,
                    stage_timings,
                    timeout=task_timeouts.get("_evaluate_completion_sync"),
                ):
                    await self.evaluate(timeout_sec=task_timeouts.get("_evaluate_completion_sync"))

            current_stage = "4_calculate_reward"
            async with self._perf_tracer.span(current_stage, cat="env"):
                async with async_timer(current_stage, stage_timings):
                    reward = await self.calculate_reward()

        except TimeoutError as e:
            logger.warning(f"⏰ Task {self.task_name} timed out at stage '{current_stage}' (timeout={task_timeouts.get('agent_astep')}s)")
            self.error_info = {
                "stage": current_stage,
                "error_message": f"TimeoutError: {e}",
            }
        except Exception as e:
            logger.error(f"\n❌ Error in task {self.task_name} at stage '{current_stage}': {e}", exc_info=True)
            self._perf_tracer.instant(
                f"error:{current_stage}", cat="env",
                args={"error_type": type(e).__name__, "error_message": str(e)},
            )
            self.error_info = {
                "stage": current_stage,
                "error_message": str(e),
            }
        finally:
            # Always close — attempt cleanup even if earlier stages failed
            if hasattr(self, 'runtime'):
                try:
                    async with self._perf_tracer.span("5_close", cat="env"):
                        async with async_timer(
                            "5_close",
                            stage_timings,
                            timeout=task_timeouts.get("_cleanup"),
                        ):
                            await self.close()
                except Exception:
                    pass

            # Save perf trace to trial directory (best-effort)
            if hasattr(self, 'output_path'):
                try:
                    self._perf_tracer.save(str(self.output_path / "perf_trace.json"))
                except Exception as _te:
                    logger.warning(f"Failed to save perf trace: {_te}")

            run_info = {
                "task_name": self.task_name,
                "uid": self.uid,
                "traj_i": self.traj_i,
                "error_info": self.error_info,
                "timings": stage_timings,
                "reward": reward,
                "evaluation": self.evaluation_results,
                "agent_summary": self.agent_summary,
            }

            # Persist run_info to trial directory so it can be collected later.
            if hasattr(self, 'output_path'):
                try:
                    import json
                    with open(self.output_path / "run_info.json", "w") as fh:
                        json.dump(run_info, fh, indent=2, default=str)
                except Exception as _ri:
                    logger.warning(f"Failed to save run_info: {_ri}")

            return run_info, reward
    
    async def evaluate(self, timeout_sec: int | None = None):
        r"""Call verifier to evaluate on the runtime.

        """
        # Leave buffer so server-side subprocess is killed before the outer
        # async_timer fires, giving time to collect output and send HTTP response.
        exec_timeout = int(timeout_sec - 30) if timeout_sec is not None else None
        self.evaluation_results = await self.verifier.verify(timeout_sec=exec_timeout)

    async def close(self) -> None:
        await self.runtime.stop()

    
    async def calculate_reward(self) -> float:
        r"""Calculate reward based on evaluation results and reward function.
        
        """
        try:
            try:
                trajectory = await load_main_trajectory(str(self.output_path / "CAMEL_LOG_DIR"))
            except FileNotFoundError:
                trajectory = None
                
            reward = await reward_factory(
                                        self.reward_fn, 
                                        evaluation_results=self.evaluation_results, 
                                        trajectory=trajectory
                                        )
            return reward
        except Exception as e:
            logger.error(f"Error in calculating reward: {e}")
            return None
    
    async def run_agent(self) -> None:
        r"""
        Run the agent in the runtime, with the toolkits initialized in the runtime.
        """
        response = await self.agent.astep(self.task.get("instruction"))
        logger.info(f"Agent final response: {response}")
        summary = dict(self.agent.meta_info_record)

        # Normalize termination reason to a plain scalar-friendly string so callers
        # can map it to one-hot metrics without depending on Enum serialization.
        termination_reason = summary.get("termination_reason")
        if termination_reason is not None:
            normalized_reason = getattr(
                termination_reason, "value", str(termination_reason)
            )
            summary["termination_reason"] = normalized_reason
            if normalized_reason in _IMPORTANT_TERMINATION_REASONS:
                summary["important_termination_reason"] = normalized_reason

        self.agent_summary = summary

    async def _reset_env(self, task: dict, uid: str) -> None:
        task_path = task.get("task_path")
        if task_path:
            self.runtime_config['task_dir'] = task_path
        self.runtime_config['session_id'] = uid
        await self._reset_runtime()
        self.output_path = self.runtime._trial_paths.trial_dir
        await self._reset_agent()

    async def reset(self) -> None:
        await self._reset_runtime()
        await self._reset_agent()

    async def _reset_runtime(self) -> None:
        r"""
        runtime config:
            task_dir: str = None,
            trial_root: str = None,
            session_id: str = None,
            environment_type: str = None,
            environment: BaseEnvironment = None,
        
        """
        if self.runtime_config.get("environment") is not None:
            self.runtime = DockerHarborRuntime(environment=self.runtime_config["environment"])
            # _task is not set when a pre-initialized environment is provided; set from task_dir
            task_dir = self.runtime_config.get('task_dir')
            if task_dir:
                from harbor.models.task.task import Task as HarborTask
                self.runtime._task = HarborTask(task_dir)
        else:
            _known = {'task_dir', 'trial_root', 'session_id', 'environment_type', 'environment', 'toolkit'}
            _extra = {k: v for k, v in self.runtime_config.items() if k not in _known}
            self.runtime = DockerHarborRuntime(
                task_dir=self.runtime_config['task_dir'],
                trial_root=self.runtime_config['trial_root'],
                session_id=self.runtime_config['session_id'],
                environment_type=self.runtime_config['environment_type'],
                **_extra,
            )
            await self.runtime.reset()

        toolkit = self.runtime_config.get("toolkit", "auto")
        await self.runtime.get_tools(toolkit=toolkit)
        self.terminal_tools = self.runtime.tools

        self.verifier = Verifier(
            task=self.runtime._task,
            trial_paths=self.runtime._trial_paths,
            environment=self.runtime.harbor_env,
        )

    async def _reset_agent(self) -> None:

        # Pop TITO keys before model creation (they're not valid model kwargs)
        tito_enabled = self.model_config.pop('tito_enabled', False)
        tito_validate = self.model_config.pop('tito_validate', False)

        model = self.model_config.get('model', None)
        # If no pre-built model is provided, create one via ModelFactory
        # by unpacking the config dict (model_platform, model_type, url, api_key, ...).
        if model is None:
            if tito_enabled:
                from seta_env.models.tito_chat_model import TITOChatModel
                tito_kwargs = {
                    k: v for k, v in self.model_config.items()
                    if k != 'model_platform'
                }
                model = TITOChatModel(
                    tito_validate=tito_validate, **tito_kwargs
                )
            else:
                from camel.models import ModelFactory
                model = ModelFactory.create(**self.model_config)

        # Enable request logging, directing output to the trial's CAMEL_LOG_DIR
        # so that calculate_reward() can load the trajectory later.
        if model is not None:
            model._log_enabled = True
            model._log_dir = str(self.output_path / "CAMEL_LOG_DIR")

        system_message = load_system_message(self.agent_config['prompt'])
        if not self.agent_config.get('thinking', True):
            system_message = system_message.rstrip() + "\n/no_think"

        token_limit = self.agent_config['max_total_tokens'] - self.agent_config.get('max_completion_tokens', 0)
        max_iteration = self.agent_config['max_iteration']

        # Resolve agent class: use registry name if provided, else default.
        AgentClass = (
            get_agent_class(self.agent_config['agent'])
            if self.agent_config.get('agent')
            else _DefaultAgent
        )

        # Notes are stored in the trial output directory so each run is isolated.
        note_toolkit = NoteTakingToolkit(working_directory=str(self.output_path / "notes"))

        tools = [t for t in self.terminal_tools+note_toolkit.get_tools() if t.get_function_name() in self.agent_config['tool_names']]

        self.agent = AgentClass(
            system_message=BaseMessage.make_assistant_message(
                role_name="Developer Agent",
                content=system_message,
            ),
            model=model,
            tools=tools,
            token_limit=token_limit,
            max_iteration=max_iteration,
            task_name=self.task_name,
            summarize_threshold=None,  # disable mid-trajectory summarization
        )

        self.agent.reset()

        # Attach the trajectory's perf tracer so the agent can record
        # per-iteration, model-request, and tool-call spans.
        if hasattr(self, '_perf_tracer'):
            self.agent._perf_tracer = self._perf_tracer
