from typing import List, Dict, Any
import json
from pathlib import Path


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


def load_prompt(stage: str, phase: str, obfuscate: bool = False, **variables) -> Dict[str, Any]:
    """
    Load and construct prompt from external files and configuration.
    
    Args:
        stage: "draft", "improve", "debug"
        phase: "planning", "coding"
        obfuscate: Whether to use obfuscated prompts
        **variables: Variables for string formatting
        
    Returns:
        Dictionary containing the constructed prompt sections
    """
    # Get the aide package directory
    aide_dir = Path(__file__).parent.parent
    prompt_dir = aide_dir / "prompts" / stage / phase
    
    # Check if prompt.json exists, if not return empty dict (fallback to old behavior)
    config_file = prompt_dir / "prompt.json"
    if not config_file.exists():
        return {}
    
    # Load prompt configuration
    config = json.loads(config_file.read_text())
    
    result = {}
    for section in config["sections"]:
        section_name = section["name"]
        
        if "source" in section:
            # Handle file source with potential obfuscation
            source_file = section["source"]
            if obfuscate and source_file.startswith("introduction"):
                source_file = source_file.replace("introduction", "introduction-obfuscate")
            
            file_path = prompt_dir / source_file
            if file_path.exists():
                if source_file.endswith('.json'):
                    # Recursively load JSON configuration
                    nested_content = _load_json_prompt_section(file_path, prompt_dir, **variables)
                    result[section_name] = nested_content
                else:
                    # Load text file
                    content = file_path.read_text().strip()
                    # Apply string formatting with provided variables
                    try:
                        result[section_name] = content.format(**variables)
                    except KeyError:
                        # If variable not found, keep original content
                        result[section_name] = content
            else:
                # If file doesn't exist, skip this section
                continue
                
        elif "value" in section:
            # Direct value with formatting
            try:
                result[section_name] = section["value"].format(**variables)
            except KeyError:
                # If variable not found, keep original value
                result[section_name] = section["value"]
    
    return result


def _load_json_prompt_section(json_file: Path, base_dir: Path, **variables) -> Dict[str, Any]:
    """
    Helper function to recursively load JSON prompt sections.
    
    Args:
        json_file: Path to the JSON file to load
        base_dir: Base directory for resolving relative paths
        **variables: Variables for string formatting
        
    Returns:
        Dictionary containing the loaded prompt sections
    """
    config = json.loads(json_file.read_text())
    result = {}
    for section in config["sections"]:
        section_name = section["name"]
        
        if "source" in section:
            source_file = section["source"]
            # Resolve relative path from the JSON file's directory
            if source_file.startswith("../"):
                file_path = json_file.parent / source_file
            else:
                file_path = base_dir / source_file
            
            if file_path.exists():
                if source_file.endswith('.json'):
                    # Recursively load nested JSON
                    nested_content = _load_json_prompt_section(file_path, base_dir, **variables)
                    result[section_name] = nested_content
                else:
                    # Load text file
                    content = file_path.read_text().strip()
                    try:
                        result[section_name] = content.format(**variables)
                    except KeyError:
                        result[section_name] = content
            else:
                continue
                
        elif "value" in section:
            try:
                result[section_name] = section["value"].format(**variables)
            except KeyError:
                result[section_name] = section["value"]
    
    return result


def load_common_prompt(filename: str, **variables) -> str:
    """
    Load a common prompt file from prompts/common/ directory.
    
    Args:
        filename: Name of the file in common directory
        **variables: Variables for string formatting
        
    Returns:
        Formatted content of the file
    """
    aide_dir = Path(__file__).parent.parent
    common_dir = aide_dir / "prompts" / "common"
    file_path = common_dir / filename
    
    if not file_path.exists():
        return ""
    
    content = file_path.read_text().strip()
    try:
        return content.format(**variables)
    except KeyError:
        return content
