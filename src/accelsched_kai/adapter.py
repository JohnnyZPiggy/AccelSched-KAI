"""Lossless mapping to the verified frozen, installed AccelSched dependency."""
import hashlib
import importlib
import importlib.util
from pathlib import Path
import numpy as np
from .schema import validate

BASELINE = "9dc0c263d2aefeb151d565166b6b950e2ac9f6a5"
SOURCE_HASHES = {
    "m0_oracle": "7243f96a0edc6fa3d439cd3cc6ae483d88255c25b7a9c6a81c515a77e90ed7d2",
    "certified_pgfp": "802b7d46084fc1c5324e1c7f44fec93a343356106c99eceb1c50063a10b57ede",
    "lp_contract": "d694498a93dd63243104dda26f6eb5348d1ebcce9ced405cfb5542c403b510a3",
    "lp_certify": "2f0819b9f54b110679efd263bfdf457d338804e1c50254f19f490e103058992e",
    "lp_backends": "f24cf7ec22e936e7668d7b4d245d0fab080234cf3975075ed9b3247abbf18d3f",
}


def frozen():
    modules = {}
    for name, expected in SOURCE_HASHES.items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError("install the pinned AccelSched dependency: " + name)
        path = Path(spec.origin).resolve()
        # Git's Windows checkout/archive newline conversion is the only normalization.
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError("frozen AccelSched source mismatch: " + name)
        modules[name] = importlib.import_module(name)
        if Path(modules[name].__file__).resolve() != path:
            raise RuntimeError("loaded module origin mismatch: " + name)
    return modules


def eligible(node, job):
    return node["id"] in job["eligible"] and (job["gpu_family"] is None or node["gpu_family"] == job["gpu_family"])


def instance(snapshot):
    s = validate(snapshot)
    labels = list(dict.fromkeys(n["shape"] for n in s["nodes"]))
    matrix = lambda jobs: np.array([[eligible(n, j) for j in jobs] for n in s["nodes"]], bool)
    a = frozen()["m0_oracle"].Instance(
        np.array([n["capacity"] for n in s["nodes"]], np.int64),
        np.array([j["demand"] for j in s["tasks"]], np.int64),
        np.array([j["value"] for j in s["tasks"]], float), matrix(s["tasks"]),
        np.array([labels.index(n["shape"]) for n in s["nodes"]], np.int64),
        np.array([j["demand"] for j in s["future"]], np.int64),
        np.array([j["weight"] for j in s["future"]], float), matrix(s["future"]))
    return a, {"nodes": [n["id"] for n in s["nodes"]], "tasks": [j["id"] for j in s["tasks"]],
               "shapes": labels, "future": [j["id"] for j in s["future"]]}
