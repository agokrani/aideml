from aide.runtime.local import LocalRuntime
from aide.runtime.modal import ModalRuntime
import modal as modal

def get_runtime(workspace_dir, cfg, task_id=None, preprocess_data=False): 
    if cfg.use_modal:
        gpu = None
        if cfg.gpu is not None: 
            try:
                gpu_cls = getattr(modal.gpu, cfg.gpu.upper())
            except:
                raise ValueError(f"GPU {cfg.gpu} not found. Please refer to modal docs: https://modal.com/docs/reference/modal.gpu for available GPUs")
            gpu_count = cfg.gpu_count if cfg.gpu_count is not None else 1
            if cfg.gpu_size is not None and cfg.gpu.upper() != 'A100':
                raise ValueError("GPU size can only be specified for A100 GPUs")
            if cfg.gpu.upper() == 'A100':
                gpu = gpu_cls(count=gpu_count, size=cfg.gpu_size) if cfg.gpu_size else gpu_cls(count=gpu_count)
            else:
                gpu = gpu_cls(count=gpu_count)
        return ModalRuntime(workspace_dir, cfg.timeout, cfg.format_tb_ipython, gpu=gpu, task_id=task_id, preprocess_data=preprocess_data)
    else:
        return LocalRuntime(workspace_dir, cfg.timeout, cfg.format_tb_ipython)
