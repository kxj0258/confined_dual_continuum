"""只解析并打印发动机实施例的通用受限空间描述。"""

from pathlib import Path

from confined_dual_continuum.workspace import load_engine_workspace


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    workspace = load_engine_workspace(root / "configs/workspace/engine_embodiment.yaml")
    print(workspace)

