import os
import time
import asyncio
import logging
import humanize
from pathlib import Path
from aide.interpreter import ExecutionResult
from aide.runtime.runtime import Runtime

logger = logging.getLogger("aide")


class LocalRuntime(Runtime):
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        # agent_file_name: str = "runfile.py",
    ):
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.process = None
        # self.process: Process = None  # type: ignore

    async def cleanup_session(self):
        self.process.kill()
        await self.process.wait()
        await self.process.communicate()

    async def run(
        self, code: str, reset_session: bool = True, is_command: bool = False
    ):
        logger.info("Executing code on local runtime")

        if reset_session:
            if self.process is not None:
                # terminate and clean up previous process
                await self.cleanup_session()
        else:
            # reset_session needs to be True on first exec
            raise NotImplementedError

        if is_command:
            command2exec = code
        else:
            with open(os.path.join(self.working_dir, "runfile.py"), "w") as f:
                f.write(code)
            command2exec = "python runfile.py"

        output: list[str] = []
        try:
            start_time = time.time()
            self.process = await asyncio.create_subprocess_shell(
                command2exec,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )
            try:
                await asyncio.wait_for(
                    self.process.wait(), timeout=self.timeout
                )
                exec_time = time.time() - start_time
                if self.process.returncode == 0:
                    stdout = await self.process.stdout.read()
                    output.append(stdout.decode())
                else:
                    stderr = await self.process.stderr.read()
                    output.append(stderr.decode())
                return ExecutionResult(output, exec_time, None, None, None)

            except asyncio.TimeoutError:
                await self.cleanup_session()
                exec_time = self.timeout
                output.append(
                    f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
                )
                return ExecutionResult(output, exec_time, "TimeoutError", {}, [])

        except Exception as e:
            return ExecutionResult(
                term_out=str(e),
                exec_time=0,
                exc_type="RuntimeError",
                exc_info={},
                exc_stack=[],
            )
