import os 
import time
import modal
import logging
import humanize
from pathlib import Path
from aide.interpreter import ExecutionResult
from aide.runtime.runtime import Runtime

logger = logging.getLogger("aide")

PYTHON_VERSION = "3.11"


# TODO: Change this to support volume and copying of inputs from the volume + Preprocess data after the copy
class ModalRuntime(Runtime):
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        gpu = None,
        task_id = None,
        preprocess_data = False,
        volume_name="agent-volume"
    ):
        self.working_dir = Path(working_dir).resolve()
        assert (
            self.working_dir.exists()
        ), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.gpu = gpu
        self.modal_working_dir = f"/workspaces/{self.working_dir.name}"
        self.task_id = task_id
        self.volume = modal.Volume.from_name(volume_name)
        
        self.preprocess_script = "/preprocess/preprocess.py"
        self.preprocess_data = preprocess_data
        if preprocess_data:
            with self.volume.batch_upload(force=True) as batch:
                batch.put_file(Path(__file__).parent.parent / "utils" / "__init__.py", self.preprocess_script)
        self.process = self._create_sandbox()  # type: ignore
        

    def _create_sandbox(self) -> modal.Sandbox:
        agent_image = modal.Image.debian_slim(
            python_version=PYTHON_VERSION
        ).pip_install_from_requirements(
            Path(__file__).parent.parent.parent / "requirements.txt"
        )
        
        self.app = modal.App.lookup(
            "aide-agent",
            create_if_missing=True
        )
        
        sandbox = modal.Sandbox.create(
            image=agent_image,
            timeout=self.timeout,  # 10 minutes
            app=self.app,
            volumes={"/vol": self.volume},
            workdir=self.modal_working_dir,
            # Modal sandboxes support GPUs!
            gpu=self.gpu,
            # you can also pass secrets here -- note that the main app's secrets are not shared
        )

        
        workdir_ls = sandbox.ls(".")
        if "input"  not in workdir_ls:
            res = sandbox.exec("cp", "-r", f"/vol/tasks/{self.task_id}", "input")
            res.wait()
            if self.preprocess_data: 
                res = sandbox.exec("python", self.preprocess_script, f"{self.modal_working_dir}/input")
                res.wait()
        
        if "working" not in workdir_ls:
            try:
                sandbox.mkdir("working")
            except modal.Error:
                logger.error(f"Error creating working directory")            
        
        if "submission" not in workdir_ls:
            try: 
                sandbox.mkdir("submission")
            except modal.Error:
                logger.log(f"Error creating submission directory")
        return sandbox

    async def cleanup_session(self, stop_app: bool = False):
        self.process.terminate()
        if stop_app:
            os.system(f"modal app stop {self.app.app_id}")

    async def run(self, code: str, reset_session: bool = False, is_command: bool = False):
        if reset_session:
            raise NotImplementedError
        
        if is_command:
            command2exec = code
        else:
            runfile_path = os.path.join(self.working_dir, "runfile.py")
            with open(runfile_path, "w") as f:
                f.write(code)
            escaped_content = code.replace("'", "'\\''")  # Proper escaping for single quotes
            self.process.exec(
                "sh", "-c", f"echo '{escaped_content}' > {os.path.basename(runfile_path)}"
            )

            command2exec = "python runfile.py"

        output: list[str] = []
        try:
            start_time = time.time()
            res = self.process.exec(*command2exec.split())
            res.wait()
           
            if res.returncode != 0:
                output.append(f"Error: {res.stderr.read()}")
            else:
                output.append(res.stdout.read())
            exec_time = time.time() - start_time
        
        except modal.exception.TimeoutError as e:
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