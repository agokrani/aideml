import copy
import json
from pathlib import Path
import dataclasses_json

from ..journal import Journal, Node
from typing import Type, TypeVar, Union


def dumps_json(obj: dataclasses_json.DataClassJsonMixin):
    """Serialize AIDE dataclasses (such as Journals) to JSON."""
    if isinstance(obj, Journal):
        obj = copy.deepcopy(obj)
        node2parent = {n.id: n.parent.id for n in obj.nodes if n.parent is not None}
        for n in obj.nodes:
            n.parent = None
            n.children = set()

    obj_dict = obj.to_dict()

    if isinstance(obj, Journal):
        obj_dict["node2parent"] = node2parent  # type: ignore
        obj_dict["__version"] = "2"

    return json.dumps(obj_dict, separators=(",", ":"))


def dump_json(obj: dataclasses_json.DataClassJsonMixin, path: Path):
    with open(path, "w") as f:
        f.write(dumps_json(obj))


G = TypeVar("G", bound=dataclasses_json.DataClassJsonMixin)


def loads_json(s: str, cls: Type[G]) -> G:
    """Deserialize JSON to AIDE dataclasses."""
    obj_dict = json.loads(s)
    obj = cls.from_dict(obj_dict)

    if isinstance(obj, Journal):
        id2nodes = {n.id: n for n in obj.nodes}
        for child_id, parent_id in obj_dict["node2parent"].items():
            id2nodes[child_id].parent = id2nodes[parent_id]
            id2nodes[child_id].__post_init__()
    return obj


def load_json(path: Path, cls: Type[G]) -> G:
    with open(path, "r") as f:
        return loads_json(f.read(), cls)


def load_code_file(path: Union[str, Path]) -> "Node":
    """
    Loads a Python file and creates a Node object from its content.

    Args:
        path (Union[str, Path]): Path to the Python file.

    Returns:
        Node: A Node object containing the file's content.

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        ValueError: If the file is not a Python file.
        IOError: If there are issues reading the file.
    """
    # Convert string path to Path object if needed
    file_path = Path(path) if isinstance(path, str) else path

    # Validate file existence and extension
    if not file_path.exists():
        raise FileNotFoundError(f"The file '{file_path}' does not exist")

    if file_path.suffix != ".py":
        raise ValueError(f"The file '{file_path}' must have a '.py' extension")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
            return Node(code=code, parent=None)
    except IOError as e:
        raise IOError(f"Error reading file '{file_path}': {str(e)}")
