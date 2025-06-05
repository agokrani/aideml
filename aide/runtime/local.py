import os
import time
import asyncio
import logging
import humanize
import shutil
from pathlib import Path
from aide.utils.execution_result import ExecutionResult
from aide.runtime.runtime import Runtime
from aide.journal import Node

logger = logging.getLogger("aide")


class LocalRuntime(Runtime):
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        debug: bool = True,
    ):
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.process = None
        # self.process: Process = None  # type: ignore
        self.agent_file_name = agent_file_name
        self.debug = debug

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
                try:
                    await self.cleanup_session()
                except ProcessLookupError:
                    logger.info("Process already terminated, skipping cleanup.")
        else:
            # reset_session needs to be True on first exec
            raise NotImplementedError

        if is_command:
            command2exec = code
        else:
            with open(os.path.join(self.working_dir, self.agent_file_name), "w") as f:
                f.write(code)
            command2exec = f"python {self.agent_file_name}"

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

                async def read_stream(stream, is_stderr=False):
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        decoded_line = line.decode().rstrip()
                        if self.debug:
                            prefix = "stderr:" if is_stderr else "stdout:"
                            print(f"{prefix} {decoded_line}", flush=True)
                        output.append(decoded_line + "\n")

                stdout_task = asyncio.create_task(read_stream(self.process.stdout))
                stderr_task = asyncio.create_task(
                    read_stream(self.process.stderr, True)
                )

                await asyncio.wait_for(self.process.wait(), timeout=self.timeout)
                await stdout_task
                await stderr_task

                exec_time = time.time() - start_time
                return ExecutionResult("".join(output), exec_time, None, None, None)

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

    async def cache_best_node(self, node: Node):
        """Cache the best node's submission and solution files for local runtime."""

        # Create best solution directory
        best_solution_dir = self.working_dir / "best_solution"
        best_solution_dir.mkdir(exist_ok=True, parents=True)

        # Create best submission directory
        best_submission_dir = self.working_dir / "best_submission"
        best_submission_dir.mkdir(exist_ok=True, parents=True)

        submission_dir = self.working_dir / "submission"
        if submission_dir.exists():
            for file_path in submission_dir.iterdir():
                if file_path.is_file():
                    shutil.copy(file_path, best_submission_dir)

        # Save solution code
        with open(best_solution_dir / "solution.py", "w") as f:
            f.write(node.code)

        # Save node ID
        with open(best_solution_dir / "node_id.txt", "w") as f:
            f.write(str(node.id))

    async def install_missing_libraries(self, missing_libraries: list[str]) -> None:
        """
        Installs missing libraries asynchronously, one by one, using pip.

        :param missing_libraries: A list of library names to install.
        :raises Exception: If any library fails to install.
        """
        import sys

        for library in missing_libraries:
            logger.info(f"Installing missing library: {library}")
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pip",
                "install",
                library,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                error_msg = stderr.decode()
                logger.error(f"Failed to install {library}. Error: {error_msg}")
                raise Exception(f"Failed to install {library}: {error_msg}")
            else:
                logger.info(f"Successfully installed {library}.")
