"""Modal app running the sudoku experiments on GPUs.

The image carries the dependencies of this optyx branch plus the JAX GPU
stack; the optyx checkout and this experiment directory are mounted so
the smoke test can contract an ``interaction.CMap`` unrolling on the GPU
through ``optyx.core.contract.contract_tensor``, and the training runs
use the batched Cotengra/JAX contraction of ``contraction.py``.

Run, from ``examples/sudoku``::

    modal run modal_app.py::smoke_test
    modal run modal_app.py --configs configs.json --out results.json
"""

import json
import pathlib
import sys

import modal

app = modal.App("optyx-sudoku")

try:
    REPO = pathlib.Path(__file__).resolve().parents[2]
except IndexError:                       # inside the container
    REPO = pathlib.Path("/root/optyx")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]==0.4.38", "optax", "cotengra", "quimb", "numpy",
        "opt_einsum", "networkx>=3.2", "sympy", "pylatexenc>=2.10",
        "perceval-quandela>=0.12,<0.13", "graphix==0.2.16",
        "pytket>=2.6.0", "pytket-pyzx>=0.35.0", "pytket-qiskit>=0.68.3",
        "git+https://github.com/discopy/discopy@88895277",
    )
    .env({"PYTHONPATH": "/root/optyx:/root/optyx/examples/sudoku"})
    .add_local_dir(
        REPO, remote_path="/root/optyx",
        ignore=[".venv", ".git", "docs", "**/__pycache__"])
)

GPU = "A100-40GB"


@app.function(image=image, gpu=GPU, timeout=1200)
def smoke_test():
    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    devices = [str(device) for device in jax.devices()]

    from optyx.channel import qubit
    from optyx.core.contract import contract_tensor
    from optyx.interaction import Box, CMap
    from optyx.qubits import Z, X, Scalar, Ket

    cnot = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)
    cmap = CMap([Box("f", qubit, qubit, cnot)], [((0, 0), (0, 1))])
    unrolled = (Ket(0) @ Ket(0)) >> cmap.unroll(1)
    network = unrolled.double().to_tensor().to_map()
    gpu_result = contract_tensor(network, backend="jax")
    cpu_result = contract_tensor(network, backend="numpy")
    assert np.allclose(
        np.asarray(gpu_result.array), np.asarray(cpu_result.array))
    return {
        "devices": devices,
        "gpu_matches_cpu": True,
        "result": np.asarray(gpu_result.array).ravel().tolist(),
    }


@app.function(image=image, gpu=GPU, timeout=6 * 3600)
def run_config(config: dict) -> dict:
    sys.path.insert(0, "/root/optyx/examples/sudoku")
    import experiment as ex
    import contraction as co

    train_cases, test_cases = ex.make_dataset()
    result, _ = co.train(config, train_cases, test_cases)
    return result


@app.local_entrypoint()
def main(configs: str = "configs.json", out: str = "results.json"):
    with open(configs) as handle:
        config_list = json.load(handle)
    results = []
    for result in run_config.map(config_list, order_outputs=False):
        results.append(result)
        with open(out, "w") as handle:
            json.dump(results, handle, indent=2)
        final = result["final"] or {}
        print(result["config"], result["n_parameters"],
              final.get("cell_accuracy"), final.get("solve_rate"),
              flush=True)
