from typing import List


def install_missing_libraries(missing_libraries: List[str]): 
    """
    Installs the missing libraries using pip.

    Args:
        missing_libraries: A list of missing library names.
    """
    import subprocess
    import sys

    for library in missing_libraries:
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])