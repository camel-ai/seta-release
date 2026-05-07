"""Env Service — runs TerminalEnvironment.step() as a remote FastAPI service.

Deploys with its own TerminalEnvConfig (agent, runtime, env settings).
Caller only sends the task payload + model URL/api_key (from ProxySession).
Config can be updated via POST /config without redeployment.

Usage:
    ENV_SERVICE_CONFIG=config.yaml MAX_SLOTS=16 ENV_SERVICE_API_KEY=dev-key \
        uvicorn seta_env.services.env_service:app --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from seta_env.environments.terminal_env import TerminalEnvironment
from seta_env.runtimes.docker_harbor_runtime import DockerHarborRuntime
from seta_env.utils.configs import (
    AgentConfig,
    EnvConfig,
    RuntimeConfig,
    TerminalEnvConfig,
    build_agent_config,
    build_env_config,
    build_model_config,
)

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────

MAX_SLOTS = int(os.environ.get("MAX_SLOTS", "16"))
API_KEY = os.environ.get("ENV_SERVICE_API_KEY", "")
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/data/harbor/dataset"))
HARBOR_ROOT = Path(os.environ.get("HARBOR_ROOT", "/tmp/harbor"))
GC_INTERVAL_SEC = 300
BUILD_TTL_SEC = 3600


def _load_terminal_env_config() -> TerminalEnvConfig:
    """Load TerminalEnvConfig from YAML file or use defaults."""
    config_path = os.environ.get("ENV_SERVICE_CONFIG", "")
    if config_path and Path(config_path).exists():
        data = yaml.safe_load(Path(config_path).read_text())
        te_data = data.get("terminal_env", data)
        # Build model config: keep as raw dict to preserve extra keys
        # like tito_enabled/tito_validate that ModelConfig doesn't have.
        # URL/api_key are overridden per /step request.
        model_cfg = None
        if "model" in te_data:
            model_data = te_data["model"]
            # Only create ModelConfig from standard fields
            model_fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
            standard = {k: v for k, v in model_data.items() if k in model_fields}
            model_cfg = ModelConfig(**standard)
        cfg = TerminalEnvConfig(
            agent=AgentConfig(**te_data["agent"]) if "agent" in te_data else AgentConfig(),
            model=model_cfg,
            runtime=RuntimeConfig(**te_data["runtime"]) if "runtime" in te_data else RuntimeConfig(),
            env=EnvConfig(**te_data["env"]) if "env" in te_data else EnvConfig(),
        )
        # Store raw model dict for extra keys (tito_enabled, tito_validate)
        cfg._raw_model_config = te_data.get("model", {})
        logger.info("Loaded config from %s", config_path)
        return cfg
    return TerminalEnvConfig(model=None)


# ── Build Gate ──────────────────────────────────────────────────────────────


@dataclass
class BuildState:
    status: Literal["building", "built", "failed"]
    event: asyncio.Event = field(default_factory=asyncio.Event)
    error: str | None = None


class BuildGate:
    """Per-task_name single-flight build coordination.

    First caller builds; subsequent callers for the same task_name wait.
    Different task_names build in parallel (independent Events).
    """

    def __init__(self):
        self._gate_lock = asyncio.Lock()
        self._registry: dict[str, BuildState] = {}
        self._timestamps: dict[str, float] = {}

    async def ensure_built(self, task_name: str, build_fn) -> None:
        async with self._gate_lock:
            if task_name not in self._registry:
                self._registry[task_name] = BuildState(status="building")
                is_builder = True
            else:
                is_builder = False
            state = self._registry[task_name]

        if is_builder:
            try:
                await build_fn()
                state.status = "built"
            except Exception as e:
                state.status = "failed"
                state.error = str(e)
                logger.error("Build failed for %s: %s", task_name, e)
            finally:
                self._timestamps[task_name] = time.monotonic()
                state.event.set()
        else:
            if state.status == "building":
                await state.event.wait()

        if state.status == "failed":
            raise RuntimeError(f"Build failed for {task_name}: {state.error}")

    def clear(self, older_than: float = BUILD_TTL_SEC) -> int:
        now = time.monotonic()
        to_remove = [
            k for k, t in self._timestamps.items() if now - t > older_than
        ]
        for k in to_remove:
            self._registry.pop(k, None)
            self._timestamps.pop(k, None)
        return len(to_remove)

    @property
    def stats(self) -> dict:
        return {
            "building": sum(1 for s in self._registry.values() if s.status == "building"),
            "built": sum(1 for s in self._registry.values() if s.status == "built"),
            "failed": sum(1 for s in self._registry.values() if s.status == "failed"),
        }


# ── Global state ────────────────────────────────────────────────────────────

_build_gate = BuildGate()
_slot_semaphore = asyncio.Semaphore(MAX_SLOTS)
_active_count = 0
_dataset_locks: dict[str, asyncio.Lock] = {}
_te_config: TerminalEnvConfig = _load_terminal_env_config()


# ── Request / Response models ───────────────────────────────────────────────


class StepRequest(BaseModel):
    """Thin request — env_service owns the TerminalEnvConfig.
    Caller only provides the task + model URL (from ProxySession)."""

    # Task to execute
    task: dict  # {"task_name", "instruction", ...}
    uid: str
    traj_i: int = 0

    # Model endpoint (from ProxySession env vars)
    model_url: str = ""      # OPENAI_BASE_URL from ProxySession
    model_api_key: str = ""  # OPENAI_API_KEY from ProxySession (= session_id)

    # Dataset info for path resolution
    dataset_name: str = ""
    task_name: str = ""

    # Trial name for organizing logs (e.g. "trial1-seta-env-v2-eval")
    trial_name: str = ""


class StepResponse(BaseModel):
    run_info: dict | None = None
    reward: float | None = None
    error: str | None = None


class SetupRequest(BaseModel):
    dataset_name: str
    hf_token: str = ""


# ── Auth ────────────────────────────────────────────────────────────────────


def _check_auth(x_api_key: str) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(403, "Invalid API key")


# ── App lifecycle ───────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    gc_task = asyncio.create_task(_gc_loop())
    logger.info(
        "env_service started: max_slots=%d, dataset_root=%s", MAX_SLOTS, DATASET_ROOT
    )
    yield
    gc_task.cancel()


app = FastAPI(title="Env Service", lifespan=lifespan)


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "max_slots": MAX_SLOTS,
        "available_slots": _slot_semaphore._value,
        "active_steps": _active_count,
        "build_gate": _build_gate.stats,
        "dataset_root": str(DATASET_ROOT),
    }


@app.get("/config")
async def get_config(x_api_key: str = Header("")):
    """Return current TerminalEnvConfig."""
    _check_auth(x_api_key)
    return {"config": asdict(_te_config)}


@app.post("/config")
async def update_config(new_config: dict, x_api_key: str = Header("")):
    """Update TerminalEnvConfig without redeployment."""
    _check_auth(x_api_key)
    global _te_config

    te_data = new_config.get("terminal_env", new_config)
    _te_config = TerminalEnvConfig(
        agent=AgentConfig(**te_data["agent"]) if "agent" in te_data else _te_config.agent,
        model=None,
        runtime=RuntimeConfig(**te_data["runtime"]) if "runtime" in te_data else _te_config.runtime,
        env=EnvConfig(**te_data["env"]) if "env" in te_data else _te_config.env,
    )
    logger.info("Config updated via POST /config")
    return {"status": "ok", "config": asdict(_te_config)}


@app.post("/step")
async def step(req: StepRequest, x_api_key: str = Header("")):
    _check_auth(x_api_key)
    global _active_count

    # 1. Resolve task_dir
    task_name = req.task_name or req.task.get("task_name", "")
    dataset_name = req.dataset_name or req.task.get("dataset_name", "")
    if dataset_name and task_name:
        task_dir = str(DATASET_ROOT / dataset_name / task_name)
    else:
        task_dir = ""

    if not task_dir or not Path(task_dir).exists():
        return StepResponse(
            error=f"task_dir not found: {task_dir}. "
            f"Run POST /setup with dataset_name={dataset_name!r} first."
        )

    # 2. Build configs from service's TerminalEnvConfig
    agent_config = build_agent_config(_te_config.agent)
    env_config = build_env_config(_te_config.runtime, _te_config.env)

    # Model config: use raw YAML dict (preserves tito_enabled, tito_validate)
    # then override URL/api_key from request (Miles Router session URL)
    raw_model = getattr(_te_config, "_raw_model_config", {})
    if raw_model:
        model_config = dict(raw_model)
    elif _te_config.model is not None:
        model_config = build_model_config(_te_config.model)
    else:
        model_config = {
            "model_platform": "sglang",
            "model_type": "",
            "model_config_dict": {"max_tokens": _te_config.agent.max_total_tokens, "stream": False},
        }
    # Override URL and api_key from request (Miles Router session URL)
    if req.model_url:
        model_config["url"] = req.model_url
    if req.model_api_key:
        model_config["api_key"] = req.model_api_key

    # Trial root: organized by trial_name
    trial_root = HARBOR_ROOT / "trials"
    if req.trial_name:
        trial_root = trial_root / req.trial_name

    runtime_config = {
        "task_dir": task_dir,
        "trial_root": str(trial_root),
        "environment_type": _te_config.runtime.env_type,
    }

    # 3. Build gate (builds also under trial_name)
    build_root = trial_root / "_builds"
    async def build_fn():
        rt = DockerHarborRuntime(
            task_dir=task_dir,
            trial_root=str(build_root),
            session_id=f"build_{task_name}",
            environment_type="docker",
        )
        try:
            await rt.build()
        finally:
            try:
                await rt.stop()
            except Exception:
                pass

    try:
        await _build_gate.ensure_built(task_name, build_fn)
    except RuntimeError as e:
        return StepResponse(error=str(e))

    # 4. Acquire slot and run
    async with _slot_semaphore:
        _active_count += 1
        try:
            task = {**req.task, "task_path": task_dir}
            te = TerminalEnvironment(
                agent_config=agent_config,
                model_config=model_config,
                runtime_config=runtime_config,
                env_config=env_config,
            )
            run_info, reward = await te.step(task, uid=req.uid, traj_i=req.traj_i)
            return StepResponse(run_info=run_info, reward=reward)
        except Exception as e:
            logger.error("step() failed for %s: %s", req.uid, e, exc_info=True)
            return StepResponse(error=str(e))
        finally:
            _active_count -= 1


@app.post("/setup")
async def setup_dataset(req: SetupRequest, x_api_key: str = Header("")):
    """Download/activate dataset. Same pattern as node_manager."""
    _check_auth(x_api_key)

    import shutil
    import tempfile

    dest = DATASET_ROOT / req.dataset_name
    if dest.exists() and any(dest.iterdir()):
        return {"status": "already_present", "path": str(dest), "success": True}

    datasets_yaml = Path(__file__).parent.parent / "dataset" / "datasets.yaml"
    if not datasets_yaml.exists():
        raise HTTPException(400, f"datasets.yaml not found at {datasets_yaml}")

    datasets_cfg = yaml.safe_load(datasets_yaml.read_text()).get("datasets", {})
    if req.dataset_name not in datasets_cfg:
        raise HTTPException(
            400,
            f"Unknown dataset: {req.dataset_name!r}. "
            f"Available: {list(datasets_cfg.keys())}",
        )

    cfg = datasets_cfg[req.dataset_name]
    repo = cfg.get("repo")
    if not repo:
        raise HTTPException(400, f"No repo URL for dataset {req.dataset_name!r}")
    subfolder = cfg.get("subfolder")

    if req.dataset_name not in _dataset_locks:
        _dataset_locks[req.dataset_name] = asyncio.Lock()

    async with _dataset_locks[req.dataset_name]:
        if dest.exists() and any(dest.iterdir()):
            return {"status": "already_present", "path": str(dest), "success": True}

        DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        clone_url = repo
        clone_env = {**os.environ}
        hf_token = req.hf_token or os.environ.get("HF_TOKEN", "")
        if hf_token and "huggingface.co" in repo:
            clone_url = repo.replace("https://", f"https://user:{hf_token}@")

        with tempfile.TemporaryDirectory() as tmpdir:
            clone_dest = f"{tmpdir}/repo"
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth=1", clone_url, clone_dest,
                env=clone_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise HTTPException(500, f"git clone failed: {stderr.decode()}")

            proc2 = await asyncio.create_subprocess_exec(
                "git", "lfs", "pull", cwd=clone_dest, env=clone_env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, lfs_err = await proc2.communicate()
            if proc2.returncode != 0:
                logger.warning("git lfs pull failed (non-fatal): %s", lfs_err.decode()[:200])

            if subfolder:
                shutil.move(f"{clone_dest}/{subfolder}", str(dest))
            else:
                shutil.move(clone_dest, str(dest))

    return {"status": "downloaded", "path": str(dest), "success": True}


@app.post("/cleanup")
async def cleanup(x_api_key: str = Header("")):
    """Full Docker cleanup: stop all, remove all, prune networks."""
    _check_auth(x_api_key)

    async def _docker(*args):
        p = await asyncio.create_subprocess_exec(
            "docker", *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await p.communicate()
        return [c for c in out.decode().strip().split("\n") if c]

    # 1. Stop all running containers
    running = await _docker("ps", "-q")
    if running:
        await (await asyncio.create_subprocess_exec(
            "docker", "stop", *running,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )).communicate()

    # 2. Remove all containers
    all_containers = await _docker("ps", "-aq")
    if all_containers:
        await (await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", *all_containers,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )).communicate()

    # 3. Prune unused networks
    await (await asyncio.create_subprocess_exec(
        "docker", "network", "prune", "-f",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )).communicate()

    # 4. Clear build gate
    _build_gate.clear(older_than=0)

    return {
        "status": "ok",
        "containers_stopped": len(running),
        "containers_removed": len(all_containers),
        "networks_pruned": True,
    }


# ── GC loop ─────────────────────────────────────────────────────────────────


async def _gc_loop():
    while True:
        await asyncio.sleep(GC_INTERVAL_SEC)
        try:
            cleared = _build_gate.clear()
            if cleared:
                logger.info("GC: cleared %d expired build entries", cleared)
        except Exception as e:
            logger.warning("GC error: %s", e)
