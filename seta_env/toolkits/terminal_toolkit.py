# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
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
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import asyncio
import atexit
import base64
import mimetypes
import re
import shlex
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

from camel.logger import get_logger
from camel.toolkits.base import BaseToolkit, manual_timeout
from camel.toolkits.function_tool import FunctionTool
from camel.utils import MCPServer
from camel.utils.tool_result import ToolResult

logger = get_logger(__name__)

# Commands that paint the screen rather than stream lines
# TODO: use capture-pane for these instead of pipe-pane + strip_ansi
SCREEN_ORIENTED_COMMANDS = {"vim", "vi", "nano", "htop", "top", "less", "man"}

TRUNCATION_THRESHOLD = 1000  # chars — keep in sync with terminal_toolkit_docker.py
TRUNCATION_HEAD = 500
TRUNCATION_TAIL = 500


# @MCPServer()
class TerminalToolkit(BaseToolkit):
    """
    Async-native terminal toolkit that manages tmux sessions inside a
    BaseEnvironment runtime. Supports blocking exec, non-blocking interactive
    sessions, and file operations.

    All public methods are async. Use asyncio.run() or await directly.

    Session state lives in self.shell_sessions keyed by agent-provided id.
    All output is logged to a single unified log file (thread-safe via asyncio.Lock).
    Long outputs are truncated for the agent but saved in full inside the runtime.

    Args:
        timeout (float): Default timeout for blocking commands in seconds.
        working_directory (Optional[str]): Working directory inside runtime;
            used as cwd for exec calls when safe_mode is True.
        session_logs_dir (str): Local directory for the unified terminal.log.
        safe_mode (bool): Restrict exec to working_directory when True.
        runtime: A BaseEnvironment instance (harbor).
    """

    def __init__(
        self,
        timeout: float = 20.0,
        working_directory: Optional[str] = None,
        session_logs_dir: str = "./session_logs",
        safe_mode: bool = False,
        runtime: Any = None,  # BaseEnvironment instance
    ):
        super().__init__()
        self.timeout = timeout
        self.working_directory = working_directory
        self.session_logs_dir = session_logs_dir
        self.safe_mode = safe_mode
        self.runtime = runtime

        # session state: id -> {tmux_name, log_path, last_offset}
        self.shell_sessions: Dict[str, Dict[str, Any]] = {}

        # unified log file — all tool calls from all sessions go here sequentially
        self._log_file = Path(session_logs_dir) / "terminal.log"
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        # asyncio.Lock guards all writes to _log_file
        self._log_lock = asyncio.Lock()

        if not self.runtime:
            raise ValueError("Runtime is required.")
        assert self.working_directory is not None, "working_directory must be set"

        # Safety net: clear session dict on exit.
        # Container teardown via stop() kills all tmux sessions inside it.
        def _sync_cleanup():
            self.shell_sessions.clear()
        atexit.register(_sync_cleanup)

    def __await__(self):
        """
        Makes the toolkit awaitable so callers do:

            toolkit = await TerminalToolkit(...)

        Runs tmux install and working-directory setup eagerly at construction
        time, raising RuntimeError immediately if tmux cannot be installed.
        """
        async def _setup():
            await self._ensure_tmux()
            await self._ensure_working_directory()
            return self
        return _setup().__await__()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    # Full ESC sequence: two-char (ESC + 0x20–0x5F) or CSI (ESC [ ... final)
    _ANSI_ESCAPE = re.compile(r'\x1B(?:[ -_]|\[[0-?]*[ -/]*[@-~])')
    # Orphaned CSI tail when a sequence is split across two reads:
    #   "\x1b"    consumed last chunk → "[?25h" arrives next  → starts with "["
    #   "\x1b["   consumed last chunk → "?25h"  arrives next  → starts with [0-?]+
    _ANSI_ORPHAN = re.compile(r'^\[[0-?]*[ -/]*[@-~]|^[0-?]+[ -/]*[@-~]')

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape sequences from raw PTY output."""
        text = self._ANSI_ESCAPE.sub('', text)
        text = self._ANSI_ORPHAN.sub('', text)
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        return text

    async def _log_entry(self, method_name: str, session_id: str, output: str, **kwargs) -> str:
        """
        Truncate output if needed (saving full copy inside runtime), write a
        structured entry to the unified terminal.log, and return the
        agent-facing (possibly truncated) output string.

        method_name: raw name of the calling method.
        session_id: session this entry belongs to.
        output: command output or status message.
        kwargs: any additional input args rendered as key: value in the log.
        """
        # --- truncation ---
        if len(output) > TRUNCATION_THRESHOLD:
            timestamp = int(time.time())
            remote_dir = "/tmp"
            remote_path = f"{remote_dir}/full_output_{session_id}_{timestamp}.txt"
            try:
                # Ensure target dir exists, then save full output
                await self.runtime.exec(f"mkdir -p {shlex.quote(remote_dir)}")
                b64_content = base64.b64encode(output.encode()).decode()
                await self.runtime.exec(
                    f"echo {shlex.quote(b64_content)} | base64 -d > {shlex.quote(remote_path)}"
                )
            except Exception as e:
                logger.warning(
                    "Failed to save full output at %s: %s", remote_path, e,
                )
                remote_path = "(failed to save full output)"
            head = output[:TRUNCATION_HEAD]
            tail = output[-TRUNCATION_TAIL:]
            output = (
                head
                + f"\n... [Output truncated. Full output saved at:"
                f" {remote_path}] ...\n"
                + tail
            )

        # --- formatting ---
        ts = datetime.now().isoformat()
        entry = f"\n{'=' * 64}\n[{ts}] {method_name} | session: {session_id}\n{'-' * 64}\n"
        for key, value in kwargs.items():
            if value is not None:
                entry += f"{key}: {value}\n"
        entry += f"\nOUTPUT:\n{output}\n{'=' * 64}\n"

        async with self._log_lock:
            async with aiofiles.open(self._log_file, 'a') as f:
                await f.write(entry)

        return output

    async def _ensure_tmux(self) -> None:
        """Install tmux if absent. Raises RuntimeError if it cannot be installed."""
        result = await self.runtime.exec("which tmux")
        if result.return_code != 0:
            for install_cmd in [
                "apt-get update -qq && apt-get install -y -qq tmux",
                "yum install -y tmux",
                "dnf install -y tmux",
                "apk add --no-cache tmux",
            ]:
                result = await self.runtime.exec(install_cmd)
                if result.return_code == 0:
                    break
            if (await self.runtime.exec("which tmux")).return_code != 0:
                raise RuntimeError(
                    "tmux is not available in the runtime and could not be installed."
                )

    async def _ensure_working_directory(self) -> None:
        """Create working_directory inside the runtime if it does not exist."""
        result = await self.runtime.exec(f"mkdir -p {shlex.quote(self.working_directory)}")
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to create working_directory '{self.working_directory}': {result.stderr}"
            )

    @manual_timeout
    async def _collect_output_until_idle(
        self,
        id: str,
        check_interval: float = 0.1,
        consecutive_empty_limit: int = 3,
        max_wait: float = 30.0,
    ) -> str:
        """
        Poll shell_view every check_interval seconds.
        Return when consecutive_empty_limit empty responses received,
        or max_wait exceeded.
        """
        output_parts = []
        consecutive_empty = 0
        start_time = time.time()

        while time.time() - start_time < max_wait:
            new_output = await self._shell_view(id)
            if new_output:
                output_parts.append(new_output)
                consecutive_empty = 0
            else:
                consecutive_empty += 1
                if consecutive_empty >= consecutive_empty_limit:
                    break
            await asyncio.sleep(check_interval)

        return "".join(output_parts)

    def _tmux_start_session(self, tmux_name: str, log_path: str) -> str:
        """
        Single shell command that creates the tmux session and wires up
        pipe-pane logging in one shot, following the terminus_2 pattern.

        Uses `script -qc` to allocate a PTY so tmux doesn't complain about
        a missing terminal, and chains new-session + pipe-pane with `\\;`.
        """
        return (
            f"export TERM=xterm-256color && "
            f"export SHELL=/bin/bash && "
            f'script -qc "'
            f"tmux new-session -d -s {shlex.quote(tmux_name)} "
            f"-x 220 -y 50 'bash' \\; "
            f"pipe-pane -t {shlex.quote(tmux_name)} "
            f"'cat > {log_path}'"
            f'" /dev/null'
        )

    async def _is_process_running(self, session_id: str) -> bool:
        """
        Check if the foreground process in the tmux pane is still running.
        Returns True if pane_current_command is not a shell (bash/sh/zsh).
        """
        tmux_name = self.shell_sessions[session_id]["tmux_name"]
        result = await self.runtime.exec(
            f"tmux display-message -t {shlex.quote(tmux_name)} -p '#{{pane_current_command}}'"
        )
        if result.return_code != 0:
            return False
        current_cmd = (result.stdout or "").strip()
        return current_cmd not in ("bash", "sh", "zsh", "")

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    @manual_timeout
    async def shell_exec(
        self,
        id: str,
        command: str,
        block: bool = True,
    ) -> str:
        r"""Execute a shell command inside a tmux session.

        The command can run in blocking mode (waits for completion) or
        non-blocking mode (runs in the background).

        block=True (default):
            Waits for the command to complete up to ``self.timeout`` seconds.
            If the command finishes in time, returns the full output.
            If timeout is exceeded, the process keeps running and the session
            is converted to a non-blocking session — use shell_view /
            shell_wait / shell_kill_process to monitor it.

        block=False:
            Starts the command in the background and returns immediately with
            initial output. Use shell_view, shell_write_to_process, shell_wait,
            and shell_kill_process to interact with the session.

        Args:
            id (str): A unique identifier for the session.
            command (str): The shell command to execute.
            block (bool): Whether to wait for completion. Defaults to True.

        Returns:
            str: If block is True, returns complete stdout/stderr or a timeout
                message with session monitoring guidance.
                If block is False, returns a start message with initial output.
        """
        tmux_name = f"terminal_{id}"
        log_path = f"{self.working_directory}/session_{id}.log"

        start_result = await self.runtime.exec(
            self._tmux_start_session(tmux_name, log_path)
        )
        if start_result.return_code != 0:
            error_msg = f"Error: Failed to start session '{id}': {start_result.stderr or start_result.stdout}"
            return await self._log_entry("shell_exec", id, error_msg, command=command, block=block)

        await self.runtime.exec(
            f"tmux send-keys -t {shlex.quote(tmux_name)} {shlex.quote(command)} Enter"
        )

        self.shell_sessions[id] = {
            "tmux_name": tmux_name,
            "log_path": log_path,
            "last_offset": 0,
        }

        if not block:
            output = await self._collect_output_until_idle(id)
            output = await self._log_entry("shell_exec", id, output, command=command, block=False)
            return f"Session '{id}' started.\n\n[Initial Output]:\n{output}"

        # block=True: poll until done or self.timeout — no restart, process
        # keeps running in the same tmux session if deadline is exceeded.
        poll_interval = 0.5
        deadline = time.time() + self.timeout

        while time.time() < deadline:
            await asyncio.sleep(poll_interval)
            if not await self._is_process_running(id):
                break
        else:
            # Timed out — leave session alive for the agent to monitor
            timeout_msg = (
                f"Command did not complete within {self.timeout} seconds. "
                f"Session '{id}' is still running.\n\n"
                f"You can use:\n"
                f"  - shell_view('{id}') - get current output\n"
                f"  - shell_wait('{id}', wait_seconds=30) - wait for completion\n"
                f"  - shell_kill_process('{id}') - terminate"
            )

            # Include partial output collected so far
            partial = await self._shell_view(id)
            if partial:
                if len(partial) > TRUNCATION_THRESHOLD:
                    partial = (
                        partial[:TRUNCATION_HEAD]
                        + "\n... [Output truncated] ...\n"
                        + partial[-TRUNCATION_TAIL:]
                    )
                timeout_msg += f"\n\n[Partial output so far]:\n{partial}"

            return await self._log_entry(
                "shell_exec", id, timeout_msg, command=command, block=True,
            )

        # Completed within timeout — collect full output and clean up
        output = await self._shell_view(id)
        await self.runtime.exec(f"tmux kill-session -t {shlex.quote(tmux_name)}")
        self.shell_sessions.pop(id)
        return await self._log_entry("shell_exec", id, output, command=command, block=block)

    async def _shell_view(self, id: str) -> str:
        r"""
        Return new output from a non-blocking session since last call.
        Uses file offset on pipe-pane log: tail -c +{offset+1} {log_path}
        Strips ANSI escape sequences and updates last_offset.
        Returns "" if no new output.

        Args:
            id (str): The unique session ID created with shell_exec(block=False).

        Returns:
            str: New output since last shell_view call, or "" if none.
        """
        if id not in self.shell_sessions:
            return f"Error: No session '{id}'."

        session = self.shell_sessions[id]
        log_path = session["log_path"]
        offset = session["last_offset"]

        result = await self.runtime.exec(
            f"tail -c +{offset + 1} {shlex.quote(log_path)}"
        )
        raw = result.stdout or ""
        stripped = self._strip_ansi(raw)

        # advance offset by raw byte length
        session["last_offset"] += len(raw.encode())


        return stripped

    @manual_timeout
    async def shell_view(self, id: str) -> str:
        r"""Retrieve new output from a non-blocking session since the last call.

        If the process has completed, appends a ``[completed]`` marker.
        If the tmux session no longer exists, cleans up and returns an error.

        Args:
            id (str): The session ID created with shell_exec(block=False)
                or converted from a timed-out blocking exec.

        Returns:
            str: New output, or empty string if nothing new.
        """
        if id not in self.shell_sessions:
            return f"Error: No session '{id}'."

        # Check if the tmux session still exists
        tmux_name = self.shell_sessions[id]["tmux_name"]
        check = await self.runtime.exec(f"tmux has-session -t {shlex.quote(tmux_name)} 2>/dev/null")
        if check.return_code != 0:
            self.shell_sessions.pop(id, None)
            return f"Error: No session '{id}'."

        try:
            stripped = await self._shell_view(id)
            running = await self._is_process_running(id)
            if not running:
                stripped += "\n[completed]"
            return await self._log_entry("shell_view", id, stripped)
        except Exception as e:
            return await self._log_entry("shell_view", id, f"Error occurred: {e}")
        
    @manual_timeout
    async def shell_write_to_process(self, id: str, command: str) -> str:
        r"""Send input to a running non-blocking session.

        Sends the given text to the process's standard input. A newline
        ``\n`` is automatically appended. Returns the output collected after
        the process becomes idle again.

        Args:
            id (str): The session ID created with shell_exec(block=False).
            command (str): The text to write to the process's stdin.

        Returns:
            str: Output collected after sending the command.
        """
        if id not in self.shell_sessions:
            return f"Error: No active session '{id}'."

        if not await self._is_process_running(id):
            return f"Error: No active session '{id}'."

        # flush pending output before sending new input
        await self._shell_view(id)

        tmux_name = self.shell_sessions[id]["tmux_name"]
        await self.runtime.exec(
            f"tmux send-keys -t {shlex.quote(tmux_name)} {shlex.quote(command)} Enter"
        )

        output = await self._collect_output_until_idle(id)
        return await self._log_entry("shell_write_to_process", id, output, command=command)

    @manual_timeout
    async def shell_wait(self, id: str, wait_seconds: float = 5.0) -> str:
        r"""Wait for a non-blocking process to produce more output or terminate.

        Polls the session every 0.5 seconds for the specified duration and
        collects all output produced during the wait. Uses wall-clock time
        so slow HTTP round-trips count against the budget.

        Args:
            id (str): The session ID created with shell_exec(block=False).
            wait_seconds (float): Maximum seconds to wait (capped at 10.0).
                Defaults to 5.0.

        Returns:
            str: All output collected during the wait period.
        """
        wait_seconds = min(wait_seconds, 10.0)

        if id not in self.shell_sessions:
            return f"Error: No session '{id}'."

        if not await self._is_process_running(id):
            return "Session is no longer running. Use shell_view to get final output."

        parts = []
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            await asyncio.sleep(0.5)
            chunk = await self._shell_view(id)
            if chunk:
                parts.append(chunk)
            # Stop early if the process has exited
            if not await self._is_process_running(id):
                # One final drain to catch any trailing output
                final = await self._shell_view(id)
                if final:
                    parts.append(final)
                break

        output = "".join(parts)
        return await self._log_entry("shell_wait", id, output, wait_seconds=wait_seconds)

    @manual_timeout
    async def shell_kill_process(self, id: str) -> str:
        r"""Terminate a running non-blocking session.

        Kills the tmux session, removes the pipe-pane log, and cleans up
        session state.

        Args:
            id (str): The session ID to terminate.

        Returns:
            str: Confirmation message.
        """
        if id not in self.shell_sessions:
            return f"Error: No active session '{id}'."

        session = self.shell_sessions.pop(id)
        tmux_name = session["tmux_name"]
        log_path = session["log_path"]

        await self.runtime.exec(f"tmux kill-session -t {shlex.quote(tmux_name)}")
        await self.runtime.exec(f"rm -f {shlex.quote(log_path)}")

        msg = f"Session '{id}' terminated."
        return await self._log_entry("shell_kill_process", id, msg)

    @manual_timeout
    async def shell_write_content_to_file(self, content: str, file_path: str) -> str:
        r"""
        Write content to a file inside the runtime.
        Writes to a local temp file and uploads via runtime.upload_file.

        Args:
            content (str): The content to write.
            file_path (str): Destination path inside the runtime.

        Returns:
            str: Success or error message.
        """
        try:
            b64_content = base64.b64encode(content.encode()).decode()
            result = await self.runtime.exec(
                f"echo {shlex.quote(b64_content)} | base64 -d > {shlex.quote(file_path)}"
            )
            if result.return_code != 0:
                raise RuntimeError(result.stderr)
            msg = f"Content written to '{file_path}'."
        except Exception as e:
            msg = f"Error writing to '{file_path}': {e}"

        return await self._log_entry("shell_write_content_to_file", "global", msg, content=content, file_path=file_path)

    async def shell_image_read(self, image_path: str) -> Any:
        """
        Read an image file from inside the runtime and return as base64 data URI.

        Args:
            image_path (str): Path to the image inside the runtime.

        Returns:
            ToolResult: Contains text description and base64 data URI image.
        """
        result = await self.runtime.exec(f"test -f {shlex.quote(image_path)}")
        if result.return_code != 0:
            return ToolResult(text=f"Error: File '{image_path}' does not exist in runtime.")

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/png"

        result = await self.runtime.exec(f"base64 -w0 {shlex.quote(image_path)}")
        if result.return_code != 0:
            return ToolResult(text=f"Error reading image '{image_path}': {result.stderr}")

        b64_data = (result.stdout or "").strip()
        data_uri = f"data:{mime_type};base64,{b64_data}"
        return ToolResult(
            text=f"Image read from '{image_path}' ({mime_type}).",
            images=[data_uri],
        )

    async def shell_ask_user_for_help(
        self,
        id: str,
        prompt: str,
        timeout: Optional[float] = None,
    ) -> str:
        r"""
        Pause and ask a human for input. Shows current session output and prompt,
        reads user input via input(), then forwards it to the process.

        Args:
            id (str): The unique session ID of the non-blocking process.
            prompt (str): The question or issue the agent needs help with.
            timeout (Optional[float]): Unused; reserved for future async input support.

        Returns:
            str: Output collected after forwarding the user's response.
        """
        if id not in self.shell_sessions:
            return f"Error: No session '{id}'."

        current_output = await self._shell_view(id)
        await self._log_entry("shell_ask_user_for_help", id, current_output, prompt=prompt)
        print(f"\n[Session '{id}' current output]:\n{current_output}")
        print(f"\n[Agent needs help]: {prompt}")
        user_input = input("Your input: ")
        return await self.shell_write_to_process(id, user_input)

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions
        in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects.
        """
        return [
            FunctionTool(self.shell_exec),
            FunctionTool(self.shell_view),
            FunctionTool(self.shell_wait),
            FunctionTool(self.shell_write_to_process),
            FunctionTool(self.shell_kill_process),
            FunctionTool(self.shell_write_content_to_file),
            FunctionTool(self.shell_image_read),
            FunctionTool(self.shell_ask_user_for_help),
        ]
