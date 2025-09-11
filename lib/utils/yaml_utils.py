from pathlib import Path
import shutil
from collections.abc import Mapping

class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)

def to_attr(x):
    if isinstance(x, Mapping):
        return AttrDict({k: to_attr(v) for k, v in x.items()})
    if isinstance(x, list):
        return [to_attr(v) for v in x]
    if isinstance(x, tuple):
        return tuple(to_attr(v) for v in x)
    return x


def update_ae_weight_path_in_yaml(yaml_path: Path, new_path: str) -> None:
    """
    Update the `paths.ae_weight_path` value inside a pipeline YAML file.

    - Preserves formatting/comments by doing a minimal line-based edit.
    - Creates a `.bak` backup next to the YAML if not already present.
    - If the key doesn't exist, inserts it under the `paths:` block.
    """
    try:
        text = yaml_path.read_text()
    except Exception as e:
        print(f"[WARN] Could not read YAML to update ae_weight_path: {e}")
        return

    lines = text.splitlines()
    out_lines = []
    in_paths = False
    paths_indent = None
    replaced = False

    for i, line in enumerate(lines):
        # Detect start of paths block
        if line.strip().startswith('paths:') and not line.strip().startswith('#'):
            in_paths = True
            paths_indent = len(line) - len(line.lstrip(' '))
            out_lines.append(line)
            continue

        if in_paths:
            cur_indent = len(line) - len(line.lstrip(' '))
            # Leaving paths block if indent decreased to <= paths_indent and line not empty
            if line.strip() and cur_indent <= paths_indent:
                in_paths = False
            else:
                # Inside paths: look for ae_weight_path
                if line.lstrip().startswith('ae_weight_path:') and not line.strip().startswith('#'):
                    indent = ' ' * (len(line) - len(line.lstrip(' ')))
                    new_line = f"{indent}ae_weight_path: \"{new_path}\""
                    out_lines.append(new_line)
                    replaced = True
                    continue

        out_lines.append(line)

    if not replaced:
        # If key not found, append under paths block (best effort)
        inserted = False
        tmp = []
        in_paths = False
        for i, line in enumerate(out_lines):
            tmp.append(line)
            if not in_paths and line.strip().startswith('paths:') and not line.strip().startswith('#'):
                in_paths = True
                paths_indent = len(line) - len(line.lstrip(' '))
                continue
            if in_paths:
                # find first line with indent <= paths_indent (end of block)
                cur_indent = len(line) - len(line.lstrip(' '))
                next_is_end = (line.strip() and cur_indent <= paths_indent and i != 0)
                if next_is_end:
                    indent = ' ' * (paths_indent + 2)
                    tmp.insert(-1, f"{indent}ae_weight_path: \"{new_path}\"")
                    inserted = True
                    in_paths = False
        out_lines = tmp
        if not inserted and in_paths:
            # paths block reached EOF; append there
            indent = ' ' * (paths_indent + 2)
            out_lines.append(f"{indent}ae_weight_path: \"{new_path}\"")

    backup = yaml_path.with_suffix(yaml_path.suffix + '.bak')
    try:
        if not backup.exists():
            shutil.copy2(yaml_path, backup)
    except Exception:
        pass
    try:
        yaml_path.write_text("\n".join(out_lines) + "\n")
        print(f"Updated {yaml_path} → paths.ae_weight_path = {new_path}")
    except Exception as e:
        print(f"[WARN] Failed to write updated YAML: {e}")

from pathlib import Path
import shutil

def update_mlp_weight_path_in_yaml(yaml_path: Path, new_path: str) -> None:
    """
    Update the `paths.mlp_weight_path` value inside a pipeline YAML file.

    - Preserves formatting/comments via minimal line-based editing.
    - Creates a `.bak` backup next to the YAML if not already present.
    - If the key doesn't exist, inserts it under the `paths:` block.
    """
    try:
        text = yaml_path.read_text()
    except Exception as e:
        print(f"[WARN] Could not read YAML to update mlp_weight_path: {e}")
        return

    lines = text.splitlines()
    out_lines = []
    in_paths = False
    paths_indent = None
    replaced = False

    for i, line in enumerate(lines):
        # Detect start of paths block (ignore commented lines)
        if line.strip().startswith('paths:') and not line.strip().startswith('#'):
            in_paths = True
            paths_indent = len(line) - len(line.lstrip(' '))
            out_lines.append(line)
            continue

        if in_paths:
            cur_indent = len(line) - len(line.lstrip(' '))
            # Leaving paths block if indent decreases to <= paths_indent and line not empty
            if line.strip() and cur_indent <= paths_indent:
                in_paths = False
            else:
                # Inside paths: look for mlp_weight_path
                if line.lstrip().startswith('mlp_weight_path:') and not line.strip().startswith('#'):
                    indent = ' ' * (len(line) - len(line.lstrip(' ')))
                    new_line = f'{indent}mlp_weight_path: "{new_path}"'
                    out_lines.append(new_line)
                    replaced = True
                    continue

        out_lines.append(line)

    if not replaced:
        # If key not found, try to insert it into the paths block (best effort)
        inserted = False
        tmp = []
        in_paths = False
        for i, line in enumerate(out_lines):
            tmp.append(line)
            if not in_paths and line.strip().startswith('paths:') and not line.strip().startswith('#'):
                in_paths = True
                paths_indent = len(line) - len(line.lstrip(' '))
                continue
            if in_paths:
                cur_indent = len(line) - len(line.lstrip(' '))
                # If we hit the end of the paths block, insert before this line
                if line.strip() and cur_indent <= paths_indent and i != 0:
                    indent = ' ' * (paths_indent + 2)
                    tmp.insert(-1, f'{indent}mlp_weight_path: "{new_path}"')
                    inserted = True
                    in_paths = False
        out_lines = tmp

        # If we were still inside paths at EOF, append there
        if not inserted and in_paths:
            indent = ' ' * (paths_indent + 2)
            out_lines.append(f'{indent}mlp_weight_path: "{new_path}"')

    # Backup and write
    backup = yaml_path.with_suffix(yaml_path.suffix + '.bak')
    try:
        if not backup.exists():
            shutil.copy2(yaml_path, backup)
    except Exception:
        pass

    try:
        yaml_path.write_text("\n".join(out_lines) + "\n")
        print(f"Updated {yaml_path} → paths.mlp_weight_path = {new_path}")
    except Exception as e:
        print(f"[WARN] Failed to write updated YAML: {e}")

