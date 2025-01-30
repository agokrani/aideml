import os
import time
import modal
import logging
import humanize
from pathlib import Path
from aide.utils.execution_result import ExecutionResult
from aide.runtime.runtime import Runtime
from aide.journal import Node

logger = logging.getLogger("aide")

PYTHON_VERSION = "3.11"


# TODO: Change this to support volume and copying of inputs from the volume + Preprocess data after the copy
class ModalRuntime(Runtime):
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        gpu=None,
        task_id=None,
        preprocess_data=False,
        volume_name="agent-volume",
        agent_file_name: str = "runfile.py",
        debug: bool = True,
    ):
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.gpu = gpu
        self.modal_working_dir = f"/vol/workspaces/{self.working_dir.name}"
        self.task_id = task_id
        self.volume = modal.Volume.from_name(volume_name)
        self.agent_file_name = agent_file_name

        self.preprocess_script = "/preprocess/preprocess.py"
        self.preprocess_data = preprocess_data
        if preprocess_data:
            with self.volume.batch_upload(force=True) as batch:
                batch.put_file(
                    Path(__file__).parent.parent / "utils" / "__init__.py",
                    self.preprocess_script,
                )
        self.process = self._create_sandbox()  # type: ignore
        self.debug = debug

    def _create_sandbox(self) -> modal.Sandbox:
        agent_image = modal.Image.debian_slim(
            python_version=PYTHON_VERSION
        ).pip_install_from_requirements(
            Path(__file__).parent.parent.parent / "requirements.txt"
        )

        self.app = modal.App.lookup("aide-agent", create_if_missing=True)

        sandbox = modal.Sandbox.create(
            image=agent_image,
            timeout=self.timeout,
            app=self.app,
            volumes={"/vol": self.volume},
            workdir=self.modal_working_dir,
            cpu=36,  # TODO: Make this configurable
            memory=36864,  # TODO: Make this configurable
            # Modal sandboxes support GPUs!
            gpu=self.gpu,
            # you can also pass secrets here -- note that the main app's secrets are not shared
        )

        workdir_ls = sandbox.ls(".")
        if "input" not in workdir_ls:
            res = sandbox.exec("cp", "-r", f"/vol/tasks/{self.task_id}", "input")
            res.wait()
            if self.preprocess_data:
                res = sandbox.exec(
                    "python", self.preprocess_script, f"{self.modal_working_dir}/input"
                )
                res.wait()

        if "working" not in workdir_ls:
            try:
                sandbox.mkdir("working")
            except modal.Error:
                logger.error("Error creating working directory")

        if "submission" not in workdir_ls:
            try:
                sandbox.mkdir("submission")
            except modal.Error:
                logger.log("Error creating submission directory")
        return sandbox

    async def cleanup_session(self, stop_app: bool = False):
        self.process.terminate()
        if stop_app:
            os.system(f"modal app stop {self.app.app_id}")

    async def run(
        self, code: str, reset_session: bool = False, is_command: bool = False
    ):
        if reset_session:
            raise NotImplementedError

        if is_command:
            command2exec = code
        else:
            runfile_path = os.path.join(self.working_dir, self.agent_file_name)
            with open(runfile_path, "w") as f:
                f.write(code)

            f = self.process.open(
                f"{self.modal_working_dir}/{self.agent_file_name}", "w"
            )
            f.write(code)
            f.close()

            command2exec = f"python {self.agent_file_name}"

        output: list[str] = []
        try:
            start_time = time.time()
            res = self.process.exec(*command2exec.split())

            for line in res.stdout:
                if self.debug:
                    print(line, flush=True)
                output.append(line)

            res.wait()

            if res.returncode != 0:
                output.append(f"Error: {res.stderr.read()}")
            # else:
            #     output.append(res.stdout.read())
            exec_time = time.time() - start_time

        except modal.exception.TimeoutError:
            exec_time = self.timeout
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
            return ExecutionResult(output, exec_time, "TimeoutError", {}, [])

        return ExecutionResult(output, exec_time, None, None, None)

    def has_submission(self):
        if "submission.csv" in self.process.ls("submission"):
            return True
        return False

    async def cache_best_node(self, node: Node):
        """Cache the best node's submission and solution files for modal runtime."""
        # Create best solution and submission directories in sandboxed env
        best_solution_dir = f"{self.modal_working_dir}/best_solution"
        best_submission_dir = f"{self.modal_working_dir}/best_submission"

        self.process.mkdir(best_solution_dir, exist_ok=True)
        self.process.mkdir(best_submission_dir, exist_ok=True)

        # Copy submission file using modal sandbox
        self.process.exec(
            "cp",
            f"{self.modal_working_dir}/submission/submission.csv",
            best_submission_dir,
        )

        # Save solution code
        f = self.process.open(f"{best_solution_dir}/solution.py", "w")
        f.write(node.code)
        f.close()

        # Save node ID
        f = self.process.open(f"{best_solution_dir}/node_id.txt", "w")
        f.write(str(node.id))
        f.close()

    async def install_missing_libraries(self, missing_libraries: list[str]):
        for library in missing_libraries:
            await self.run(
                f"pip install {library}", reset_session=False, is_command=True
            )

    def remove_previous_submissions_directory(self):
        self.process.rm("submission", recursive=True)
        self.process.mkdir("submission")
