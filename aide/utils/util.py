from typing import List


def install_missing_libraries(
    missing_libraries: List[str],
    callback_manager=None,
    exec_callback=None,
    use_modal=False,
):
    """
    Installs the missing libraries using pip.

    Args:
        missing_libraries: A list of missing library names.
    """

    if not use_modal:
        import subprocess
        import sys

        for library in missing_libraries:
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])
