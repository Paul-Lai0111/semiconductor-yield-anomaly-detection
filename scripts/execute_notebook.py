"""Execute main.ipynb with active Python 3.12; save outputs only after success."""

import os
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use the locked Python 3.12 environment: uv sync --locked")
    python = Path(sys.executable)

    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    notebook_path = root / "main.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    kernel = KernelManager(kernel_name="python3")
    # Override the launch command without installing a persistent kernelspec.
    kernel.kernel_spec.argv = [
        str(python),
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook,
        km=kernel,
        resources={"metadata": {"path": str(root)}},
        startup_timeout=300,
        timeout=300,
        allow_errors=False,
        record_timing=False,
        on_cell_execute=lambda cell, cell_index: print(
            f"Executing cell {cell_index + 1}", flush=True
        ),
    )
    with tempfile.TemporaryDirectory(prefix="secom-notebook-") as runtime:
        # Personal IPython startup files must not change the reproducible notebook.
        environment = {**os.environ, "IPYTHONDIR": runtime, "JUPYTER_CONFIG_DIR": runtime}
        client.execute(cleanup_kc=True, env=environment)
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    print(f"Executed and saved {notebook_path}")


if __name__ == "__main__":
    main()
